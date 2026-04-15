#define TRY_OPTIONAL(expr) \
    if (auto res{expr}) {  \
        return res;        \
    }
