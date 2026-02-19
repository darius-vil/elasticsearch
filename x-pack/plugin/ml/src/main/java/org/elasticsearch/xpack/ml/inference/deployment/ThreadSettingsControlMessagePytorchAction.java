/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ml.inference.deployment;

import org.elasticsearch.action.ActionListener;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.threadpool.ThreadPool;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xpack.ml.inference.pytorch.results.PyTorchThreadSettingsResponse;
import org.elasticsearch.xpack.ml.inference.pytorch.results.ThreadSettings;

import java.io.IOException;

public class ThreadSettingsControlMessagePytorchAction extends AbstractControlMessagePyTorchAction<
    ThreadSettings,
    PyTorchThreadSettingsResponse> {
    private final int numAllocationThreads;

    ThreadSettingsControlMessagePytorchAction(
        String deploymentId,
        long requestId,
        int numAllocationThreads,
        TimeValue timeout,
        DeploymentManager.ProcessContext processContext,
        ThreadPool threadPool,
        ActionListener<ThreadSettings> listener
    ) {
        super(deploymentId, requestId, timeout, processContext, threadPool, listener);
        this.numAllocationThreads = numAllocationThreads;
    }

    @Override
    int controlOrdinal() {
        return ControlMessageTypes.AllocationThreads.ordinal();
    }

    @Override
    void writeMessage(XContentBuilder builder) throws IOException {
        builder.field("num_allocations", numAllocationThreads);
    }

    @Override
    Class<PyTorchThreadSettingsResponse> expectedResultType() {
        return PyTorchThreadSettingsResponse.class;
    }

    @Override
    ThreadSettings getResult(PyTorchThreadSettingsResponse result) {
        return result.threadSettings();
    }
}
