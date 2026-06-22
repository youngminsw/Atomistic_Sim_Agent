OAuth core attribution

This package ports the OAuth callback/PKCE/provider-interface pattern from
Yeachan-Heo/gajae-code, which is distributed under the MIT License.

Source reference:
https://github.com/Yeachan-Heo/gajae-code

Upstream license notice observed in the referenced repository:

MIT License

Copyright (c) 2025 Mario Zechner
Copyright (c) 2025-2026 Can Boluk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

The port keeps the same high-level OAuth architecture while adapting the runtime
boundary for this project:
- TypeScript model/auth gateway code is isolated from the Python simulation agent.
- Provider-specific endpoints are intentionally separated from the common OAuth flow.
- Python OpenAI Agents SDK remains the production agent runtime.
