#pragma once

#include <boost/core/null_deleter.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/sources/logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/sources/severity_feature.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <iomanip>
#include <iostream>

#define LOG_DEBUG(log) BOOST_LOG_SEV(log, gpu_planning::log_severity::DEBUG)
#define LOG_INFO(log) BOOST_LOG_SEV(log, gpu_planning::log_severity::INFO)
#define LOG_WARN(log) BOOST_LOG_SEV(log, gpu_planning::log_severity::WARNING)
#define LOG_ERROR(log) BOOST_LOG_SEV(log, gpu_planning::log_severity::ERROR)

namespace gpu_planning {

enum class log_severity { DEBUG, INFO, WARNING, ERROR };

std::ostream& operator<<(std::ostream& os, log_severity level);

using logger = boost::log::sources::severity_logger<log_severity>;

void init_logging(bool verbose);
logger create_logger();

}  // namespace gpu_planning
