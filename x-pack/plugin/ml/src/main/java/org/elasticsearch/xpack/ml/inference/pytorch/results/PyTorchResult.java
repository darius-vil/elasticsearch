/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ml.inference.pytorch.results;

import org.elasticsearch.xcontent.ConstructingObjectParser;
import org.elasticsearch.xcontent.ParseField;
import org.elasticsearch.xcontent.ToXContentObject;

/**
 * Sealed interface capturing the possible output types from the pytorch process.
 * The JSON type is determined by which fields are present; the parser inspects
 * the populated field and returns the matching subtype.
 */
public sealed interface PyTorchResult extends ToXContentObject permits PyTorchInferenceResponse, PyTorchThreadSettingsResponse,
    PyTorchAckResponse, PyTorchErrorResponse {

    String requestId();

    ParseField REQUEST_ID = new ParseField("request_id");
    ParseField CACHE_HIT = new ParseField("cache_hit");
    ParseField TIME_MS = new ParseField("time_ms");
    ParseField RESULT = new ParseField("result");
    ParseField THREAD_SETTINGS = new ParseField("thread_settings");
    ParseField ACK = new ParseField("ack");

    ConstructingObjectParser<PyTorchResult, Void> PARSER = createParser();

    static PyTorchResult fromParsed(
        String requestId,
        Boolean isCacheHit,
        Long timeMs,
        PyTorchInferenceResult inferenceResult,
        ThreadSettings threadSettings,
        AckResult ackResult,
        ErrorResult errorResult
    ) {
        if (errorResult != null) return new PyTorchErrorResponse(requestId, errorResult);
        if (inferenceResult != null) return new PyTorchInferenceResponse(requestId, isCacheHit, timeMs, inferenceResult);
        if (threadSettings != null) return new PyTorchThreadSettingsResponse(requestId, threadSettings);
        if (ackResult != null) return new PyTorchAckResponse(requestId, ackResult);
        return new PyTorchErrorResponse(requestId, new ErrorResult("unknown result type"));
    }

    private static ConstructingObjectParser<PyTorchResult, Void> createParser() {
        ConstructingObjectParser<PyTorchResult, Void> parser = new ConstructingObjectParser<>(
            "pytorch_result",
            a -> fromParsed(
                (String) a[0],
                (Boolean) a[1],
                (Long) a[2],
                (PyTorchInferenceResult) a[3],
                (ThreadSettings) a[4],
                (AckResult) a[5],
                (ErrorResult) a[6]
            )
        );
        parser.declareString(ConstructingObjectParser.constructorArg(), REQUEST_ID);
        parser.declareBoolean(ConstructingObjectParser.optionalConstructorArg(), CACHE_HIT);
        parser.declareLong(ConstructingObjectParser.optionalConstructorArg(), TIME_MS);
        parser.declareObject(ConstructingObjectParser.optionalConstructorArg(), PyTorchInferenceResult.PARSER, RESULT);
        parser.declareObject(ConstructingObjectParser.optionalConstructorArg(), ThreadSettings.PARSER, THREAD_SETTINGS);
        parser.declareObject(ConstructingObjectParser.optionalConstructorArg(), AckResult.PARSER, ACK);
        parser.declareObject(ConstructingObjectParser.optionalConstructorArg(), ErrorResult.PARSER, ErrorResult.ERROR);
        return parser;
    }
}
