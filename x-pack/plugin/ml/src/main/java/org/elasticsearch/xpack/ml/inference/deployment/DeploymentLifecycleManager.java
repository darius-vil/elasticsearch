/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ml.inference.deployment;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.ResourceNotFoundException;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.action.search.SearchRequest;
import org.elasticsearch.action.search.TransportSearchAction;
import org.elasticsearch.action.support.master.AcknowledgedResponse;
import org.elasticsearch.client.internal.Client;
import org.elasticsearch.common.util.concurrent.AbstractRunnable;
import org.elasticsearch.common.xcontent.LoggingDeprecationHandler;
import org.elasticsearch.common.xcontent.XContentHelper;
import org.elasticsearch.index.query.IdsQueryBuilder;
import org.elasticsearch.search.SearchHit;
import org.elasticsearch.threadpool.ThreadPool;
import org.elasticsearch.xcontent.NamedXContentRegistry;
import org.elasticsearch.xcontent.XContentParser;
import org.elasticsearch.xcontent.XContentType;
import org.elasticsearch.xpack.core.ml.action.GetTrainedModelsAction;
import org.elasticsearch.xpack.core.ml.inference.TrainedModelConfig;
import org.elasticsearch.xpack.core.ml.inference.trainedmodel.NlpConfig;
import org.elasticsearch.xpack.core.ml.inference.trainedmodel.VocabularyConfig;
import org.elasticsearch.xpack.core.ml.job.messages.Messages;
import org.elasticsearch.xpack.core.ml.utils.ExceptionsHelper;
import org.elasticsearch.xpack.core.ml.utils.MlPlatformArchitecturesUtil;
import org.elasticsearch.xpack.ml.MachineLearning;
import org.elasticsearch.xpack.ml.inference.nlp.NlpTask;
import org.elasticsearch.xpack.ml.inference.nlp.Vocabulary;
import org.elasticsearch.xpack.ml.inference.pytorch.process.PyTorchProcessFactory;
import org.elasticsearch.xpack.ml.notifications.InferenceAuditor;

import java.io.IOException;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutorService;
import java.util.function.Consumer;

import static org.elasticsearch.core.Strings.format;
import static org.elasticsearch.xpack.core.ClientHelper.ML_ORIGIN;
import static org.elasticsearch.xpack.core.ClientHelper.executeAsyncWithOrigin;
import static org.elasticsearch.xpack.ml.MachineLearning.UTILITY_THREAD_POOL_NAME;

/**
 * Manages the lifecycle of model deployments: starting, stopping, stats collection,
 * and the central process-context registry. Extracted from {@code DeploymentManager}
 * as part of the class decomposition.
 * <p>
 * Inference dispatch and control-message dispatch are handled separately by
 * {@link ModelInferenceHandler} and {@link ModelControlHandler}.
 */
public class DeploymentLifecycleManager {

    private static final Logger logger = LogManager.getLogger(DeploymentLifecycleManager.class);

    private final Client client;
    private final NamedXContentRegistry xContentRegistry;
    private final PyTorchProcessFactory pyTorchProcessFactory;
    private final ExecutorService executorServiceForDeployment;
    private final ExecutorService executorServiceForProcess;
    private final ThreadPool threadPool;
    private final InferenceAuditor inferenceAuditor;
    private final ConcurrentMap<Long, ProcessContext> processContextByAllocation = new ConcurrentHashMap<>();
    private final int maxProcesses;

    public DeploymentLifecycleManager(
        Client client,
        NamedXContentRegistry xContentRegistry,
        ThreadPool threadPool,
        PyTorchProcessFactory pyTorchProcessFactory,
        int maxProcesses,
        InferenceAuditor inferenceAuditor
    ) {
        this.client = Objects.requireNonNull(client);
        this.xContentRegistry = Objects.requireNonNull(xContentRegistry);
        this.pyTorchProcessFactory = Objects.requireNonNull(pyTorchProcessFactory);
        this.threadPool = Objects.requireNonNull(threadPool);
        this.inferenceAuditor = Objects.requireNonNull(inferenceAuditor);
        this.executorServiceForDeployment = threadPool.executor(UTILITY_THREAD_POOL_NAME);
        this.executorServiceForProcess = threadPool.executor(MachineLearning.NATIVE_INFERENCE_COMMS_THREAD_POOL_NAME);
        this.maxProcesses = maxProcesses;
    }

    // ------- Stats -------

    public Optional<ModelStats> getStats(TrainedModelDeploymentTask task) {
        return Optional.ofNullable(processContextByAllocation.get(task.getId())).map(processContext -> {
            var stats = processContext.getResultProcessor().getResultStats();
            var recentStats = stats.recentStats();
            return new ModelStats(
                processContext.startTime,
                stats.timingStats().getCount(),
                stats.timingStats().getAverage(),
                stats.timingStatsExcludingCacheHits().getAverage(),
                stats.lastUsed(),
                processContext.priorityProcessWorker.queueSize() + stats.numberOfPendingResults(),
                stats.errorCount(),
                stats.cacheHitCount(),
                processContext.rejectedExecutionCount.intValue(),
                processContext.timeoutCount.intValue(),
                processContext.numThreadsPerAllocation,
                processContext.numAllocations,
                stats.peakThroughput(),
                recentStats.requestsProcessed(),
                recentStats.avgInferenceTime(),
                recentStats.cacheHitCount()
            );
        });
    }

    // ------- Process registry -------

    ProcessContext addProcessContext(Long id, ProcessContext processContext) {
        return processContextByAllocation.putIfAbsent(id, processContext);
    }

    /**
     * Looks up the {@link ProcessContext} for the given task, reporting errors via the supplied consumer.
     * Returns {@code null} if the task is stopped or no context exists, after invoking the error consumer.
     * <p>
     * Package-private so that {@code ModelInferenceHandler} and {@code ModelControlHandler} can share
     * this validation logic.
     */
    ProcessContext getProcessContext(TrainedModelDeploymentTask task, Consumer<Exception> errorConsumer) {
        if (task.isStopped()) {
            errorConsumer.accept(
                ExceptionsHelper.conflictStatusException(
                    "[{}] is stopping or stopped due to [{}]",
                    task.getDeploymentId(),
                    task.stoppedReason().orElse("")
                )
            );
            return null;
        }

        ProcessContext processContext = processContextByAllocation.get(task.getId());
        if (processContext == null) {
            errorConsumer.accept(ExceptionsHelper.conflictStatusException("[{}] process context missing", task.getDeploymentId()));
            return null;
        }

        return processContext;
    }

    // ------- Lifecycle: start -------

    public void startDeployment(TrainedModelDeploymentTask task, ActionListener<TrainedModelDeploymentTask> finalListener) {
        startDeployment(task, null, finalListener);
    }

    public void startDeployment(
        TrainedModelDeploymentTask task,
        Integer startsCount,
        ActionListener<TrainedModelDeploymentTask> finalListener
    ) {
        logger.info("[{}] Starting model deployment of model [{}]", task.getDeploymentId(), task.getModelId());

        if (processContextByAllocation.size() >= maxProcesses) {
            finalListener.onFailure(
                ExceptionsHelper.serverError(
                    "[{}] Could not start inference process as the node reached the max number [{}] of processes",
                    task.getDeploymentId(),
                    maxProcesses
                )
            );
            return;
        }

        ProcessContext processContext = new ProcessContext(
            task,
            startsCount,
            client,
            executorServiceForProcess,
            executorServiceForDeployment,
            xContentRegistry,
            pyTorchProcessFactory,
            threadPool,
            inferenceAuditor,
            processContextByAllocation::remove,
            (t, sc, listener) -> startDeployment(t, sc, listener)
        );
        if (addProcessContext(task.getId(), processContext) != null) {
            finalListener.onFailure(
                ExceptionsHelper.serverError("[{}] Could not create inference process as one already exists", task.getDeploymentId())
            );
            return;
        }

        ActionListener<TrainedModelDeploymentTask> failedDeploymentListener = ActionListener.wrap(finalListener::onResponse, failure -> {
            ProcessContext failedContext = processContextByAllocation.remove(task.getId());
            if (failedContext != null) {
                failedContext.forcefullyStopProcess();
            }
            finalListener.onFailure(failure);
        });

        ActionListener<Boolean> modelLoadedListener = ActionListener.wrap(success -> {
            executorServiceForProcess.execute(() -> processContext.getResultProcessor().process(processContext.getProcess().get()));
            finalListener.onResponse(task);
        }, failedDeploymentListener::onFailure);

        ActionListener<TrainedModelConfig> getVerifiedModel = ActionListener.wrap((modelConfig) -> {
            processContext.getModelInput().set(modelConfig.getInput());
            processContext.getPrefixStrings().set(modelConfig.getPrefixStrings());

            if (modelConfig.getInferenceConfig() instanceof NlpConfig nlpConfig) {
                task.init(nlpConfig);

                SearchRequest searchRequest = vocabSearchRequest(nlpConfig.getVocabularyConfig(), modelConfig.getModelId());
                executeAsyncWithOrigin(
                    client,
                    ML_ORIGIN,
                    TransportSearchAction.TYPE,
                    searchRequest,
                    ActionListener.wrap(searchVocabResponse -> {
                        if (searchVocabResponse.getHits().getHits().length == 0) {
                            failedDeploymentListener.onFailure(
                                new ResourceNotFoundException(
                                    Messages.getMessage(
                                        Messages.VOCABULARY_NOT_FOUND,
                                        modelConfig.getModelId(),
                                        VocabularyConfig.docId(modelConfig.getModelId())
                                    )
                                )
                            );
                            return;
                        }

                        Vocabulary vocabulary = parseVocabularyDocLeniently(searchVocabResponse.getHits().getAt(0));
                        NlpTask nlpTask = new NlpTask(nlpConfig, vocabulary);
                        NlpTask.Processor processor = nlpTask.createProcessor();
                        processContext.getNlpTaskProcessor().set(processor);
                        executorServiceForDeployment.execute(new AbstractRunnable() {

                            @Override
                            public void onFailure(Exception e) {
                                failedDeploymentListener.onFailure(e);
                            }

                            @Override
                            protected void doRun() {
                                processContext.startAndLoad(modelConfig.getLocation(), modelLoadedListener);
                            }
                        });
                    }, failedDeploymentListener::onFailure)
                );
            } else {
                failedDeploymentListener.onFailure(
                    new IllegalArgumentException(
                        format(
                            "[%s] must be a pytorch model; found inference config of kind [%s]",
                            modelConfig.getModelId(),
                            modelConfig.getInferenceConfig().getWriteableName()
                        )
                    )
                );
            }
        }, failedDeploymentListener::onFailure);

        ActionListener<GetTrainedModelsAction.Response> verifyModelAndClusterArchitecturesListener = ActionListener.wrap(
            getModelResponse -> {
                assert getModelResponse.getResources().results().size() == 1;
                TrainedModelConfig modelConfig = getModelResponse.getResources().results().get(0);

                verifyMlNodesAndModelArchitectures(modelConfig, client, threadPool, getVerifiedModel);

            },
            failedDeploymentListener::onFailure
        );

        executeAsyncWithOrigin(
            client,
            ML_ORIGIN,
            GetTrainedModelsAction.INSTANCE,
            new GetTrainedModelsAction.Request(task.getParams().getModelId()),
            verifyModelAndClusterArchitecturesListener
        );
    }

    // ------- Lifecycle: stop -------

    public void stopDeployment(TrainedModelDeploymentTask task) {
        ProcessContext processContext = processContextByAllocation.remove(task.getId());
        if (processContext != null) {
            logger.info("[{}] Stopping deployment, reason [{}]", task.getDeploymentId(), task.stoppedReason().orElse("unknown"));
            processContext.forcefullyStopProcess();
        } else {
            logger.warn("[{}] No process context to stop", task.getDeploymentId());
        }
    }

    public void stopAfterCompletingPendingWork(TrainedModelDeploymentTask task, ActionListener<AcknowledgedResponse> listener) {
        ProcessContext processContext = processContextByAllocation.remove(task.getId());
        if (processContext != null) {
            logger.info(
                "[{}] Stopping deployment after completing pending tasks, reason [{}]",
                task.getDeploymentId(),
                task.stoppedReason().orElse("unknown")
            );
            processContext.stopProcessAfterCompletingPendingWork(listener);
        } else {
            logger.warn("[{}] No process context to stop gracefully", task.getDeploymentId());
        }
    }

    // ------- Vocabulary / architecture helpers -------

    void verifyMlNodesAndModelArchitectures(
        TrainedModelConfig configToReturn,
        Client client,
        ThreadPool threadPool,
        ActionListener<TrainedModelConfig> configToReturnListener
    ) {
        ActionListener<TrainedModelConfig> verifyConfigListener = new ActionListener<TrainedModelConfig>() {
            @Override
            public void onResponse(TrainedModelConfig config) {
                assert Objects.equals(config, configToReturn);
                configToReturnListener.onResponse(configToReturn);
            }

            @Override
            public void onFailure(Exception e) {
                configToReturnListener.onFailure(e);
            }
        };

        callVerifyMlNodesAndModelArchitectures(configToReturn, verifyConfigListener, client, threadPool);
    }

    void callVerifyMlNodesAndModelArchitectures(
        TrainedModelConfig configToReturn,
        ActionListener<TrainedModelConfig> configToReturnListener,
        Client client,
        ThreadPool threadPool
    ) {
        MlPlatformArchitecturesUtil.verifyMlNodesAndModelArchitectures(
            configToReturnListener,
            client,
            threadPool.executor(MachineLearning.UTILITY_THREAD_POOL_NAME),
            configToReturn
        );
    }

    private SearchRequest vocabSearchRequest(VocabularyConfig vocabularyConfig, String modelId) {
        return client.prepareSearch(vocabularyConfig.getIndex())
            .setQuery(new IdsQueryBuilder().addIds(VocabularyConfig.docId(modelId)))
            .setSize(1)
            .setTrackTotalHits(false)
            .request();
    }

    Vocabulary parseVocabularyDocLeniently(SearchHit hit) throws IOException {
        try (
            XContentParser parser = XContentHelper.createParserNotCompressed(
                LoggingDeprecationHandler.XCONTENT_PARSER_CONFIG.withRegistry(xContentRegistry),
                hit.getSourceRef(),
                XContentType.JSON
            )
        ) {
            return Vocabulary.PARSER.apply(parser, null);
        } catch (IOException e) {
            logger.error(() -> "failed to parse trained model vocabulary [" + hit.getId() + "]", e);
            throw e;
        }
    }
}
