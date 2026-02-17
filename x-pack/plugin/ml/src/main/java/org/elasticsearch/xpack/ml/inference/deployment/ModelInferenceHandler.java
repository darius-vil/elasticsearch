/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ml.inference.deployment;

import org.elasticsearch.action.ActionListener;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.inference.InferenceResults;
import org.elasticsearch.tasks.CancellableTask;
import org.elasticsearch.threadpool.ThreadPool;
import org.elasticsearch.xpack.core.ml.inference.TrainedModelPrefixStrings;
import org.elasticsearch.xpack.core.ml.inference.trainedmodel.InferenceConfig;
import org.elasticsearch.xpack.ml.inference.pytorch.PriorityProcessWorkerExecutorService;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Handles inference dispatch for deployed models. Creates {@link InferencePyTorchAction}
 * instances and submits them to the per-deployment priority queue via {@link ProcessContext}.
 * <p>
 * Extracted from {@code DeploymentManager} as part of the class decomposition.
 * The {@link ProcessContext} is obtained from {@link DeploymentLifecycleManager#getProcessContext}.
 */
public class ModelInferenceHandler {

    private static final AtomicLong requestIdCounter = new AtomicLong(1);

    private final DeploymentLifecycleManager lifecycleManager;
    private final ThreadPool threadPool;

    public ModelInferenceHandler(DeploymentLifecycleManager lifecycleManager, ThreadPool threadPool) {
        this.lifecycleManager = Objects.requireNonNull(lifecycleManager);
        this.threadPool = Objects.requireNonNull(threadPool);
    }

    public void infer(
        TrainedModelDeploymentTask task,
        InferenceConfig config,
        NlpInferenceInput input,
        boolean skipQueue,
        TimeValue timeout,
        TrainedModelPrefixStrings.PrefixType prefixType,
        CancellableTask parentActionTask,
        boolean chunkResponse,
        ActionListener<InferenceResults> listener
    ) {
        var processContext = lifecycleManager.getProcessContext(task, listener::onFailure);
        if (processContext == null) {
            // error reporting handled in the call to getProcessContext
            return;
        }

        final long requestId = requestIdCounter.getAndIncrement();
        InferencePyTorchAction inferenceAction = new InferencePyTorchAction(
            task.getDeploymentId(),
            requestId,
            timeout,
            processContext,
            config,
            input,
            prefixType,
            threadPool,
            parentActionTask,
            chunkResponse,
            listener
        );

        PriorityProcessWorkerExecutorService.RequestPriority priority = skipQueue
            ? PriorityProcessWorkerExecutorService.RequestPriority.HIGH
            : PriorityProcessWorkerExecutorService.RequestPriority.NORMAL;

        processContext.executePyTorchAction(priority, inferenceAction);
    }
}
