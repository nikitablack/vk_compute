#include <expected>

#define __CONCAT__(a, b) a##b

#define __TRY_EXPECTED_VOID_IMPL(expr, uniq)                         \
    {                                                                \
        auto __CONCAT__(_tmp_, uniq){expr};                          \
        if (!__CONCAT__(_tmp_, uniq).has_value()) {                  \
            return std::unexpected{__CONCAT__(_tmp_, uniq).error()}; \
        }                                                            \
    }

#define TRY_EXPECTED_VOID(expr) __TRY_EXPECTED_VOID_IMPL(expr, __COUNTER__)

#define __TRY_EXPECTED_IMPL(var, expr, uniq)                     \
    auto __CONCAT__(_tmp_, uniq){expr};                          \
    if (!__CONCAT__(_tmp_, uniq).has_value()) {                  \
        return std::unexpected{__CONCAT__(_tmp_, uniq).error()}; \
    }                                                            \
    var = std::move(__CONCAT__(_tmp_, uniq).value());

#define TRY_EXPECTED(var, expr) __TRY_EXPECTED_IMPL(var, expr, __COUNTER__)

#define __TRY_EXPECTED_REF_IMPL(var, expr, uniq)                 \
    auto __CONCAT__(_tmp_, uniq){expr};                          \
    if (!__CONCAT__(_tmp_, uniq).has_value()) {                  \
        return std::unexpected{__CONCAT__(_tmp_, uniq).error()}; \
    }                                                            \
    var = __CONCAT__(_tmp_, uniq).value().get();

#define TRY_EXPECTED_REF(var, expr) __TRY_EXPECTED_REF_IMPL(var, expr, __COUNTER__)
