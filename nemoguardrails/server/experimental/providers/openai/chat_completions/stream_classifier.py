# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Staged classifier metadata for the OpenAI Chat Completions stream projection."""

from nemoguardrails.server.experimental.provider.payload import ProjectionFieldCoverage
from nemoguardrails.server.experimental.provider.stream import (
    StreamCapabilityProfile,
    StreamEventRole,
    StreamProjectionContract,
    StreamShapeCoverage,
)
from nemoguardrails.server.experimental.provider.stream_classifier import (
    StreamClassifierDefinition,
    StreamEventRule,
    build_stream_classifier,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.stream_projection import (
    ChatCompletionsStreamErrorProjection,
    ChatCompletionsStreamPayloadProjection,
)

PROVIDER_DOCUMENT_URL = (
    "https://github.com/openai/openai-openapi/blob/df63773f69f542ef875b9f00c3837c25ba5f4f2a/openapi.yaml"
)
PROVIDER_REVISION = "df63773f69f542ef875b9f00c3837c25ba5f4f2a"
PROVIDER_DOCUMENT_VERSION = "2.3.0"
PROVIDER_DOCUMENT_SHA256 = "f2dae1a9aced09b91310db89edda51bf1e36ecbfb05230c3a50b239c07708469"
PROJECTION_ID = "openai.chat_completions.stream.text.v1"
CAPABILITY_PROFILE = "single_text_delta.v1"
STREAM_SOURCE_SCHEMA = "CreateChatCompletionStreamResponse"
REJECTED_PROVIDER_SCHEMAS = ()
GUARDED_SHAPES = frozenset(["chat.completion.chunk:content"])
SNAPSHOT_SHAPES = frozenset([])
OPAQUE_SHAPES = frozenset(["[DONE]", "chat.completion.chunk:metadata"])
PROVIDER_ERROR_SHAPES = frozenset(["error"])
STREAM_FIELDS = (
    frozenset(["choices"]),
    frozenset(["object"]),
    frozenset(["created", "id", "model", "moderation", "obfuscation", "service_tier", "system_fingerprint", "usage"]),
)

STREAM_CONTRACT = StreamProjectionContract(
    projection_id=PROJECTION_ID,
    profile=StreamCapabilityProfile.SINGLE_TEXT_DELTA_V1,
    shapes=StreamShapeCoverage(
        guarded_shapes=GUARDED_SHAPES,
        snapshot_shapes=SNAPSHOT_SHAPES,
        opaque_shapes=OPAQUE_SHAPES,
        provider_error_shapes=PROVIDER_ERROR_SHAPES,
    ),
    fields=ProjectionFieldCoverage(
        guarded_fields=STREAM_FIELDS[0],
        constrained_fields=STREAM_FIELDS[1],
        opaque_fields=STREAM_FIELDS[2],
    ),
)

STREAM_CLASSIFIER = build_stream_classifier(
    StreamClassifierDefinition(
        subject=PROJECTION_ID,
        contract=STREAM_CONTRACT,
        non_data_shape="[DONE]",
        sentinels=((b"[DONE]", "[DONE]"),),
        rules=(
            StreamEventRule(
                shape="chat.completion.chunk:content",
                model=ChatCompletionsStreamPayloadProjection,
                role=StreamEventRole.GUARDED_TEXT,
                match=(("object", "chat.completion.chunk"),),
                text_path=("choices", 0, "delta", "content"),
                missing_text_role=StreamEventRole.OPAQUE_METADATA,
                missing_text_shape="chat.completion.chunk:metadata",
            ),
            StreamEventRule(
                shape="error",
                model=ChatCompletionsStreamErrorProjection,
                role=StreamEventRole.PROVIDER_ERROR,
                required_fields=frozenset({"error"}),
            ),
        ),
    )
)
