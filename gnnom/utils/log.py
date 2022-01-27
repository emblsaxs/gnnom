import time, sys, inspect
import logging

logger = logging.getLogger(__name__)

log_and_raise_error = lambda logger, message: log(logger, message, lvl="ERROR", exception=RuntimeError, rollback=2)
log_warning         = lambda logger, message: log(logger, message, lvl="WARNING", exception=None, rollback=2)
log_info            = lambda logger, message: log(logger, message, lvl="INFO", exception=None, rollback=2)
log_debug           = lambda logger, message: log(logger, message, lvl="DEBUG", exception=None, rollback=2)


def log(logger, message, lvl, exception=None, rollback=1):
    logcalls = {"ERROR": logger.error,
                "WARNING": logger.warning,
                "DEBUG": logger.debug,
                "INFO": logger.info}
    if lvl not in logcalls:
        print(f"{lvl} is an invalid logger level.")
        sys.exit(1)
    logcall = logcalls[lvl]

    if (logger.getEffectiveLevel() >= logging.INFO) or rollback is None:
        # Short output
        msg = f"{message}"
    else:
        # Detailed output only in debug mode
        func = inspect.currentframe()
        for r in range(rollback):
            # Rolling back in the stack, otherwise it would be this function
            func = func.f_back
        code = func.f_code
        msg = f"{message} => in {func.f_globals['__name__']} function {code.co_name} => {code.co_filename}," \
              f" line: {code.co_firstlineno}"

    logcall(f"{lvl}: {msg}")
    if exception is not None:
        raise exception(message)


def log_execution_time(logger):
    def st_time(func):
        def st_func(*args, **keyArgs):
            t1 = time.monotonic()
            r = func(*args, **keyArgs)
            t2 = time.monotonic()
            try:
                filename = inspect.getsourcefile(func)
                module = inspect.getmodule(func)
                line = inspect.getsourcelines(func)[1]
                loc = "\'%s\' [%s:%i]" % (func.__name__,
                                          filename,
                                          line)
            except TypeError:
                loc = "\'%s\'" % func.__name__
            msg = "Execution time = %.4f sec\n\t=> in function %s" % (t2 - t1, loc)
            log(logger, msg, "DEBUG", exception=None, rollback=None)
            return r

        return st_func

    return st_time
