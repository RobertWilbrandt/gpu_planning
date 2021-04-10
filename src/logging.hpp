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

std::ostream& operator<<(std::ostream& os, log_severity level) {
  switch (level) {
    case log_severity::DEBUG:
      return os << "DEBUG";
    case log_severity::INFO:
      return os << "INFO";
    case log_severity::WARNING:
      return os << "WARN";
    case log_severity::ERROR:
      return os << "ERROR";
    default:
      return os << "UNKNOWN";
  }
}

using logger = boost::log::sources::severity_logger<log_severity>;

void init_logging() {
  // Create synchronous text sink
  namespace sinks = boost::log::sinks;
  using text_sink = sinks::synchronous_sink<sinks::text_ostream_backend>;
  boost::shared_ptr<text_sink> sink = boost::make_shared<text_sink>();

  // Add cout stream to sink
  boost::shared_ptr<std::ostream> stream(&std::cout, boost::null_deleter{});
  sink->locked_backend()->add_stream(stream);

  // Set formatting of sink
  namespace expr = boost::log::expressions;
  sink->set_formatter(expr::stream << '[' << std::setw(5)
                                   << expr::attr<log_severity>("Severity")
                                   << "] " << expr::smessage);

  // Register sink
  boost::log::core::get()->add_sink(sink);
}

logger create_logger() {
  namespace keywords = boost::log::keywords;
  return logger{keywords::severity = log_severity::INFO};
}

}  // namespace gpu_planning
