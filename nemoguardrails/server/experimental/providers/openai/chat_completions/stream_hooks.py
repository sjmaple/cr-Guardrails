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

"""Bind OpenAI Chat Completions protocol hooks to the staged classifier."""

from nemoguardrails.server.experimental.provider.sse import ServerSentEvent
from nemoguardrails.server.experimental.provider.stream import GuardedStreamEvent


class ChatCompletionsStreamHooks:
    """Implement Chat Completions protocol state and native error framing."""

    def observe_event(self, event: ServerSentEvent, classified: GuardedStreamEvent) -> None:
        """Accept every event classified by the staged stream semantics."""

    def validate_end_of_stream(self) -> None:
        """Accept the end of a Chat Completions stream."""

    def encode_error(self, rendered_body: bytes) -> tuple[bytes, bytes]:
        """Encode an OpenAI-compatible error and terminal chat event."""

        return b"data: " + rendered_body + b"\n\n", b"data: [DONE]\n\n"
