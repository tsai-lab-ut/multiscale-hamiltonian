"""
Logging utils.
"""

using LoggingExtras, Dates

const date_format = "yyyy-mm-dd HH:MM:SS"

timestamp_logger(logger) = TransformerLogger(logger) do log
    merge(log, (; message = "$(Dates.format(now(), date_format)) $(log.message)"))
end

function get_default_logger(dir)
    mkpath(dir)
    filepath = joinpath(dir, "run.log")
    logger = timestamp_logger(MinLevelLogger(FileLogger(filepath), Logging.Info))
    return logger
end