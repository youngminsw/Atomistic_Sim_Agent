from __future__ import annotations


class SchemaValidationError(ValueError):
    pass


class ProviderConfigPolicyError(SchemaValidationError):
    pass
