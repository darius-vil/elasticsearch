/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ml.inference.pytorch.results;

import org.elasticsearch.core.Nullable;
import org.elasticsearch.xcontent.XContentBuilder;

import java.io.IOException;

public record PyTorchInferenceResponse(
    String requestId,
    @Nullable Boolean isCacheHit,
    @Nullable Long timeMs,
    @Nullable PyTorchInferenceResult inferenceResult
) implements PyTorchResult {

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field(REQUEST_ID.getPreferredName(), requestId);
        if (isCacheHit != null) {
            builder.field(CACHE_HIT.getPreferredName(), isCacheHit);
        }
        if (timeMs != null) {
            builder.field(TIME_MS.getPreferredName(), timeMs);
        }
        if (inferenceResult != null) {
            builder.field(RESULT.getPreferredName(), inferenceResult);
        }
        builder.endObject();
        return builder;
    }
}
