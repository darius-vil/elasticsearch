/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ml.inference.pytorch.results;

import org.elasticsearch.xcontent.XContentBuilder;

import java.io.IOException;

public record PyTorchThreadSettingsResponse(String requestId, ThreadSettings threadSettings) implements PyTorchResult {

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field(REQUEST_ID.getPreferredName(), requestId);
        builder.field(THREAD_SETTINGS.getPreferredName(), threadSettings);
        builder.endObject();
        return builder;
    }
}
