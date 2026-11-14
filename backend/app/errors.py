class DomainError(Exception):
    def __init__(self, message: str, status: int = 400, code: str = "domain_error"):
        super().__init__(message)
        self.status = status
        self.code = code


class ProviderError(Exception):
    def __init__(self, message: str, code: str = "provider_error", retryable: bool = False):
        super().__init__(message)
        self.code, self.retryable = code, retryable


class InterruptedRun(Exception):
    pass


class BudgetExhausted(Exception):
    def __init__(self, category="model_calls"):
        self.category = category
        super().__init__(category)
