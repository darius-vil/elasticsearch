/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ml.inference.deployment;

import org.elasticsearch.action.ActionListener;
import org.elasticsearch.action.support.master.AcknowledgedResponse;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.threadpool.ThreadPool;
import org.elasticsearch.xpack.ml.inference.pytorch.PriorityProcessWorkerExecutorService;
import org.elasticsearch.xpack.ml.inference.pytorch.results.ThreadSettings;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Handles control message dispatch for deployed models: cache clearing and
 * allocation thread updates. Creates the corresponding control-message actions
 * and submits them at {@link PriorityProcessWorkerExecutorService.RequestPriority#HIGHEST}
 * priority via {@link ProcessContext}.
 * <p>
 * Extracted from {@code DeploymentManager} as part of the class decomposition.
 * The {@link ProcessContext} is obtained from {@link DeploymentLifecycleManager#getProcessContext}.
 */
public class ModelControlHandler {

    private static final AtomicLong requestIdCounter = new AtomicLong(1);

    private final DeploymentLifecycleManager lifecycleManager;
    private final ThreadPool threadPool;

    public ModelControlHandler(DeploymentLifecycleManager lifecycleManager, ThreadPool threadPool) {
        this.lifecycleManager = Objects.requireNonNull(lifecycleManager);
        this.threadPool = Objects.requireNonNull(threadPool);
    }

    public void clearCache(TrainedModelDeploymentTask task, TimeValue timeout, ActionListener<AcknowledgedResponse> listener) {
        var processContext = lifecycleManager.getProcessContext(task, listener::onFailure);
        if (processContext == null) {
            // error reporting handled in the call to getProcessContext
            return;
        }

        final long requestId = requestIdCounter.getAndIncrement();
        ClearCacheControlMessagePytorchAction controlMessageAction = new ClearCacheControlMessagePytorchAction(
            task.getDeploymentId(),
            requestId,
            timeout,
            processContext,
            threadPool,
            listener.delegateFailureAndWrap((l, b) -> l.onResponse(AcknowledgedResponse.TRUE))
        );

        processContext.executePyTorchAction(PriorityProcessWorkerExecutorService.RequestPriority.HIGHEST, controlMessageAction);
    }

    public void updateNumAllocations(
        TrainedModelDeploymentTask task,
        int numAllocationThreads,
        TimeValue timeout,
        ActionListener<ThreadSettings> listener
    ) {
        var processContext = lifecycleManager.getProcessContext(task, listener::onFailure);
        if (processContext == null) {
            // error reporting handled in the call to getProcessContext
            return;
        }

        final long requestId = requestIdCounter.getAndIncrement();
        ThreadSettingsControlMessagePytorchAction controlMessageAction = new ThreadSettingsControlMessagePytorchAction(
            task.getDeploymentId(),
            requestId,
            numAllocationThreads,
            timeout,
            processContext,
            threadPool,
            listener
        );

        processContext.executePyTorchAction(PriorityProcessWorkerExecutorService.RequestPriority.HIGHEST, controlMessageAction);
    }
}
