/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ml.inference.deployment;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.util.SetOnce;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.action.support.ListenerTimeouts;
import org.elasticsearch.action.support.master.AcknowledgedResponse;
import org.elasticsearch.client.internal.Client;
import org.elasticsearch.common.Strings;
import org.elasticsearch.common.util.concurrent.EsRejectedExecutionException;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.threadpool.ThreadPool;
import org.elasticsearch.xcontent.NamedXContentRegistry;
import org.elasticsearch.xpack.core.ml.inference.TrainedModelInput;
import org.elasticsearch.xpack.core.ml.inference.TrainedModelPrefixStrings;
import org.elasticsearch.xpack.core.ml.inference.trainedmodel.IndexLocation;
import org.elasticsearch.xpack.core.ml.inference.trainedmodel.TrainedModelLocation;
import org.elasticsearch.xpack.ml.MachineLearning;
import org.elasticsearch.xpack.ml.inference.nlp.NlpTask;
import org.elasticsearch.xpack.ml.inference.pytorch.PriorityProcessWorkerExecutorService;
import org.elasticsearch.xpack.ml.inference.pytorch.process.PyTorchProcess;
import org.elasticsearch.xpack.ml.inference.pytorch.process.PyTorchProcessFactory;
import org.elasticsearch.xpack.ml.inference.pytorch.process.PyTorchResultProcessor;
import org.elasticsearch.xpack.ml.inference.pytorch.process.PyTorchStateStreamer;
import org.elasticsearch.xpack.ml.notifications.InferenceAuditor;

import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

import static org.elasticsearch.core.Strings.format;
import static org.elasticsearch.xpack.ml.MachineLearning.UTILITY_THREAD_POOL_NAME;

/**
 * Holds all per-deployment state: the PyTorch process, NLP processor, result processor,
 * priority queue worker, etc. Extracted from {@code DeploymentManager} inner class to a
 * top-level package-private class with explicit dependency injection.
 */
class ProcessContext {

    private static final Logger logger = LogManager.getLogger(ProcessContext.class);
    private static final String PROCESS_NAME = "inference process";
    private static final TimeValue COMPLETION_TIMEOUT = TimeValue.timeValueMinutes(3);
    static final int NUM_RESTART_ATTEMPTS = 3;
    private static final TimeValue WORKER_QUEUE_COMPLETION_TIMEOUT = TimeValue.timeValueMinutes(5);

    private final TrainedModelDeploymentTask task;
    private final SetOnce<PyTorchProcess> process = new SetOnce<>();
    private final SetOnce<NlpTask.Processor> nlpTaskProcessor = new SetOnce<>();
    private final SetOnce<TrainedModelInput> modelInput = new SetOnce<>();
    private final SetOnce<TrainedModelPrefixStrings> prefixes = new SetOnce<>();
    private final PyTorchResultProcessor resultProcessor;
    private final PyTorchStateStreamer stateStreamer;
    final PriorityProcessWorkerExecutorService priorityProcessWorker;
    final AtomicInteger rejectedExecutionCount = new AtomicInteger();
    final AtomicInteger timeoutCount = new AtomicInteger();
    private final AtomicInteger startsCount = new AtomicInteger();
    volatile Instant startTime;
    volatile Integer numThreadsPerAllocation;
    volatile Integer numAllocations;
    private volatile boolean isStopped;

    // Injected dependencies (previously accessed via outer class)
    private final ExecutorService executorServiceForProcess;
    private final ExecutorService executorServiceForDeployment;
    private final PyTorchProcessFactory pyTorchProcessFactory;
    private final ThreadPool threadPool;
    private final InferenceAuditor inferenceAuditor;

    // Injected callbacks to break circular dependency with DeploymentManager
    private final Consumer<Long> removeProcessContext;
    private final RestartDeploymentHandler restartDeployment;

    /**
     * Callback interface for restarting a deployment after a crash.
     */
    @FunctionalInterface
    interface RestartDeploymentHandler {
        void restart(TrainedModelDeploymentTask task, int startsCount, ActionListener<TrainedModelDeploymentTask> listener);
    }

    ProcessContext(
        TrainedModelDeploymentTask task,
        Integer startsCount,
        Client client,
        ExecutorService executorServiceForProcess,
        ExecutorService executorServiceForDeployment,
        NamedXContentRegistry xContentRegistry,
        PyTorchProcessFactory pyTorchProcessFactory,
        ThreadPool threadPool,
        InferenceAuditor inferenceAuditor,
        Consumer<Long> removeProcessContext,
        RestartDeploymentHandler restartDeployment
    ) {
        this.task = Objects.requireNonNull(task);
        this.executorServiceForProcess = Objects.requireNonNull(executorServiceForProcess);
        this.executorServiceForDeployment = Objects.requireNonNull(executorServiceForDeployment);
        this.pyTorchProcessFactory = Objects.requireNonNull(pyTorchProcessFactory);
        this.threadPool = Objects.requireNonNull(threadPool);
        this.inferenceAuditor = Objects.requireNonNull(inferenceAuditor);
        this.removeProcessContext = Objects.requireNonNull(removeProcessContext);
        this.restartDeployment = Objects.requireNonNull(restartDeployment);

        resultProcessor = new PyTorchResultProcessor(task.getDeploymentId(), threadSettings -> {
            this.numThreadsPerAllocation = threadSettings.numThreadsPerAllocation();
            this.numAllocations = threadSettings.numAllocations();
        });
        // We want to use the inference thread pool to load the model as it is a possibly long operation
        // and knowing it is an inference thread would enable better understanding during debugging.
        // Even though we account for 3 threads per process in the thread pool, loading the model
        // happens before we start input/output so it should be ok to use a thread from that pool for loading
        // the model.
        this.stateStreamer = new PyTorchStateStreamer(client, executorServiceForProcess, xContentRegistry);
        this.priorityProcessWorker = new PriorityProcessWorkerExecutorService(
            threadPool.getThreadContext(),
            PROCESS_NAME,
            task.getParams().getQueueCapacity()
        );
        this.startsCount.set(startsCount == null ? 1 : startsCount);
    }

    PyTorchResultProcessor getResultProcessor() {
        return resultProcessor;
    }

    synchronized void startAndLoad(TrainedModelLocation modelLocation, ActionListener<Boolean> loadedListener) {
        assert Thread.currentThread().getName().contains(UTILITY_THREAD_POOL_NAME)
            : format("Must execute from [%s] but thread is [%s]", UTILITY_THREAD_POOL_NAME, Thread.currentThread().getName());

        if (isStopped) {
            logger.debug("[{}] model stopped before it is started", task.getDeploymentId());
            loadedListener.onFailure(new IllegalArgumentException("model stopped before it is started"));
            return;
        }

        logger.debug("[{}] start and load", task.getDeploymentId());
        process.set(
            pyTorchProcessFactory.createProcess(
                task,
                executorServiceForProcess,
                () -> resultProcessor.awaitCompletion(COMPLETION_TIMEOUT.getMinutes(), TimeUnit.MINUTES),
                onProcessCrashHandleRestarts(this.startsCount, task.getDeploymentId())
            )
        );
        startTime = Instant.now();
        logger.debug("[{}] process started", task.getDeploymentId());
        try {
            loadModel(modelLocation, loadedListener.delegateFailureAndWrap((delegate, success) -> {
                if (isStopped) {
                    logger.debug("[{}] model loaded but process is stopped", task.getDeploymentId());
                    killProcessIfPresent();
                    delegate.onFailure(new IllegalStateException("model loaded but process is stopped"));
                    return;
                }

                logger.debug("[{}] model loaded, starting priority process worker thread", task.getDeploymentId());
                startPriorityProcessWorker();
                delegate.onResponse(success);
            }));
        } catch (Exception e) {
            loadedListener.onFailure(e);
        }
    }

    private Consumer<String> onProcessCrashHandleRestarts(AtomicInteger startsCount, String deploymentId) {
        return (reason) -> {
            if (isThisProcessOlderThan1Day()) {
                startsCount.set(1);
                {
                    String logMessage = "["
                        + task.getDeploymentId()
                        + "] inference process crashed due to reason ["
                        + reason
                        + "]. This process was started more than 24 hours ago; "
                        + "the starts count is reset to 1.";
                    logger.error(logMessage);
                }
            } else {
                logger.error("[{}] inference process crashed due to reason [{}]", task.getDeploymentId(), reason);
            }

            removeProcessContext.accept(task.getId());
            isStopped = true;
            resultProcessor.signalIntentToStop();
            stateStreamer.cancel();

            if (startsCount.get() <= NUM_RESTART_ATTEMPTS) {
                {
                    String logAndAuditMessage = "Inference process ["
                        + task.getDeploymentId()
                        + "] failed due to ["
                        + reason
                        + "]. This is the ["
                        + startsCount.get()
                        + "] failure in 24 hours, and the process will be restarted.";
                    logger.info(logAndAuditMessage);
                    threadPool.executor(MachineLearning.UTILITY_THREAD_POOL_NAME)
                        .execute(() -> inferenceAuditor.warning(deploymentId, logAndAuditMessage));
                }
                priorityProcessWorker.shutdownNow();
                ActionListener<TrainedModelDeploymentTask> errorListener = ActionListener.wrap(
                    (trainedModelDeploymentTask -> {
                        logger.debug("Completed restart of inference process, the [{}] start", startsCount);
                    }),
                    (e) -> finishClosingProcess(
                        startsCount,
                        "Failed to restart inference process because of error [" + e.getMessage() + "]",
                        deploymentId
                    )
                );

                restartDeployment.restart(task, startsCount.incrementAndGet(), errorListener);
            } else {
                finishClosingProcess(startsCount, reason, deploymentId);
            }
        };
    }

    private boolean isThisProcessOlderThan1Day() {
        return startTime.isBefore(Instant.now().minus(Duration.ofDays(1)));
    }

    private void finishClosingProcess(AtomicInteger startsCount, String reason, String deploymentId) {
        String logAndAuditMessage = "["
            + task.getDeploymentId()
            + "] inference process failed after ["
            + startsCount.get()
            + "] starts in 24 hours, not restarting again.";
        logger.warn(logAndAuditMessage);
        threadPool.executor(MachineLearning.UTILITY_THREAD_POOL_NAME)
            .execute(() -> inferenceAuditor.error(deploymentId, logAndAuditMessage));
        priorityProcessWorker.shutdownNowWithError(new IllegalStateException(reason));
        if (nlpTaskProcessor.get() != null) {
            nlpTaskProcessor.get().close();
        }
        task.setFailed("inference process crashed due to reason [" + reason + "]");
    }

    void startPriorityProcessWorker() {
        executorServiceForProcess.submit(priorityProcessWorker::start);
    }

    synchronized void forcefullyStopProcess() {
        logger.debug(() -> format("[%s] Forcefully stopping process", task.getDeploymentId()));
        prepareInternalStateForShutdown();

        priorityProcessWorker.shutdownNow();
        try {
            // wait for any currently executing work to finish
            if (priorityProcessWorker.awaitTermination(10L, TimeUnit.SECONDS)) {
                priorityProcessWorker.notifyQueueRunnables();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            logger.info(Strings.format("[%s] Interrupted waiting for process worker after shutdownNow", PROCESS_NAME));
        }

        killProcessIfPresent();
        closeNlpTaskProcessor();
    }

    private void prepareInternalStateForShutdown() {
        isStopped = true;
        resultProcessor.signalIntentToStop();
        stateStreamer.cancel();
    }

    private void killProcessIfPresent() {
        try {
            if (process.get() == null) {
                return;
            }
            process.get().kill(true);
        } catch (IOException e) {
            logger.error(() -> "[" + task.getDeploymentId() + "] Failed to kill process", e);
        }
    }

    private void closeNlpTaskProcessor() {
        if (nlpTaskProcessor.get() != null) {
            nlpTaskProcessor.get().close();
        }
    }

    synchronized void stopProcessAfterCompletingPendingWork(ActionListener<AcknowledgedResponse> listener) {
        logger.debug(() -> format("[%s] Stopping process after completing its pending work", task.getDeploymentId()));
        prepareInternalStateForShutdown();

        // Waiting for the process worker to finish the pending work could
        // take a long time. To avoid blocking the calling thread register
        // a function with the process worker queue that is called when the
        // worker queue is finished. Then proceed to closing the native process
        // and wait for all results to be processed, the second part can be
        // done synchronously as it is not expected to take long.

        // This listener closes the native process and waits for the results
        // after the worker queue has finished
        var closeProcessListener = listener.delegateFailureAndWrap((l, r) -> {
            // process worker stopped within allotted time, close process
            closeProcessAndWaitForResultProcessor();
            closeNlpTaskProcessor();
            l.onResponse(AcknowledgedResponse.TRUE);
        });

        // Timeout listener waits
        var listenWithTimeout = ListenerTimeouts.wrapWithTimeout(
            threadPool,
            WORKER_QUEUE_COMPLETION_TIMEOUT,
            threadPool.executor(MachineLearning.UTILITY_THREAD_POOL_NAME),
            closeProcessListener,
            (l) -> {
                // Stopping the process worker timed out, kill the process
                logger.warn(
                    format("[%s] Timed out waiting for process worker to complete, forcing a shutdown", task.getDeploymentId())
                );
                forcefullyStopProcess();
                l.onResponse(AcknowledgedResponse.FALSE);
            }
        );

        priorityProcessWorker.shutdownWithCallback(() -> listenWithTimeout.onResponse(AcknowledgedResponse.TRUE));
    }

    private void closeProcessAndWaitForResultProcessor() {
        try {
            closeProcessIfPresent();
            resultProcessor.awaitCompletion(COMPLETION_TIMEOUT.getMinutes(), TimeUnit.MINUTES);
        } catch (TimeoutException e) {
            logger.warn(format("[%s] Timed out waiting for results processor to stop", task.getDeploymentId()), e);
        }
    }

    private void closeProcessIfPresent() {
        try {
            if (process.get() == null) {
                return;
            }

            process.get().close();
        } catch (IOException e) {
            logger.error(format("[%s] Failed to stop process gracefully, attempting to kill it", task.getDeploymentId()), e);
            killProcessIfPresent();
        }
    }

    void loadModel(TrainedModelLocation modelLocation, ActionListener<Boolean> listener) {
        if (isStopped) {
            listener.onFailure(new IllegalArgumentException("Process has stopped, model loading canceled"));
            return;
        }
        if (modelLocation instanceof IndexLocation indexLocation) {
            // Loading the model happens on the inference thread pool but when we get the callback
            // we need to return to the utility thread pool to avoid leaking the thread we used.
            process.get()
                .loadModel(
                    task.getParams().getModelId(),
                    indexLocation.getIndexName(),
                    stateStreamer,
                    ActionListener.wrap(
                        r -> executorServiceForDeployment.submit(() -> listener.onResponse(r)),
                        e -> executorServiceForDeployment.submit(() -> listener.onFailure(e))
                    )
                );
        } else {
            listener.onFailure(
                new IllegalStateException("unsupported trained model location [" + modelLocation.getClass().getSimpleName() + "]")
            );
        }
    }

    /**
     * Submits a PyTorch action to this context's priority process worker, handling rejected execution.
     */
    void executePyTorchAction(PriorityProcessWorkerExecutorService.RequestPriority priority, AbstractPyTorchAction<?> action) {
        try {
            getPriorityProcessWorker().executeWithPriority(action, priority, action.getRequestId());
        } catch (EsRejectedExecutionException e) {
            getRejectedExecutionCount().incrementAndGet();
            action.onFailure(e);
        } catch (Exception e) {
            action.onFailure(e);
        }
    }

    // accessor used for mocking in tests
    AtomicInteger getTimeoutCount() {
        return timeoutCount;
    }

    // accessor used for mocking in tests
    PriorityProcessWorkerExecutorService getPriorityProcessWorker() {
        return priorityProcessWorker;
    }

    // accessor used for mocking in tests
    AtomicInteger getRejectedExecutionCount() {
        return rejectedExecutionCount;
    }

    SetOnce<TrainedModelInput> getModelInput() {
        return modelInput;
    }

    SetOnce<PyTorchProcess> getProcess() {
        return process;
    }

    SetOnce<NlpTask.Processor> getNlpTaskProcessor() {
        return nlpTaskProcessor;
    }

    SetOnce<TrainedModelPrefixStrings> getPrefixStrings() {
        return prefixes;
    }
}
