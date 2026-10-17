
#include "common/logging/log.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fmt/format.h>
#include <iterator>
#include <memory>
#include <mutex>
#include <spdlog/formatter.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <string_view>

namespace {

class RawFormatter final: public spdlog::formatter {
public:
	void format(const spdlog::details::log_msg& msg, spdlog::memory_buf_t& dest) override {
		const auto timestamp = std::chrono::time_point_cast<std::chrono::milliseconds>(msg.time);
		const auto time_value = std::chrono::system_clock::to_time_t(msg.time);
		const auto millis    = timestamp.time_since_epoch().count() % 1000;
		std::tm local_time {};
#if defined(_WIN32)
		localtime_s(&local_time, &time_value);
#else
		localtime_r(&time_value, &local_time);
#endif
		char timestamp_text[32] = "0000-00-00 00:00:00";
		std::strftime(timestamp_text, sizeof(timestamp_text), "%Y-%m-%d %H:%M:%S", &local_time);
		std::string_view level = "info";
		switch (msg.level) {
			case spdlog::level::trace: level = "trace"; break;
			case spdlog::level::debug: level = "debug"; break;
			case spdlog::level::info: level = "info"; break;
			case spdlog::level::warn: level = "warning"; break;
			case spdlog::level::err: level = "error"; break;
			case spdlog::level::critical: level = "critical"; break;
			default: level = "off"; break;
		}
		fmt::format_to(std::back_inserter(dest), "[{}.{:03}] [{}] ", timestamp_text, millis, level);
		const std::string_view payload(msg.payload.data(), msg.payload.size());
		dest.append(payload.data(), payload.data() + payload.size());
	}

	std::unique_ptr<spdlog::formatter> clone() const override {
		return std::make_unique<RawFormatter>();
	}
};

std::shared_ptr<spdlog::logger> MakeLogger(std::string name, spdlog::sink_ptr sink) {
	sink->set_formatter(std::make_unique<RawFormatter>());

	auto logger = std::make_shared<spdlog::logger>(std::move(name), std::move(sink));
	logger->set_level(spdlog::level::trace);
	return logger;
}

std::shared_ptr<spdlog::logger> MakeFileLogger(std::string                  name,
                                               const std::filesystem::path& path) {
	const auto parent = path.parent_path();
	if (!parent.empty()) {
		std::filesystem::create_directories(parent);
	}

	auto sink =
	    std::make_shared<spdlog::sinks::basic_file_sink_mt>(path.native(), true);
	return MakeLogger(std::move(name), std::move(sink));
}

std::filesystem::path WarningLogPath(const std::filesystem::path& path) {
	auto filename = path.filename();
	const auto extension = filename.extension();
	filename.replace_extension();
	filename += std::filesystem::path("-warnings");
	filename += extension;
	auto warning_path = path;
	warning_path.replace_filename(filename);
	return warning_path;
}

spdlog::level::level_enum ToSpdlogLevel(Log::Level level) {
	switch (level) {
		case Log::Level::Trace: return spdlog::level::trace;
		case Log::Level::Debug: return spdlog::level::debug;
		case Log::Level::Info: return spdlog::level::info;
		case Log::Level::Warning: return spdlog::level::warn;
		case Log::Level::Error: return spdlog::level::err;
		case Log::Level::Critical: return spdlog::level::critical;
	}
	return spdlog::level::info;
}

Log::Level LevelForStyle(fmt::text_style style) {
	if (style == Log::Color::Red || style == Log::Color::BrightRed) return Log::Level::Error;
	if (style == Log::Color::Yellow || style == Log::Color::BrightYellow) return Log::Level::Warning;
	return Log::Level::Info;
}

static bool HasStyle(fmt::text_style style) {
	return style != fmt::text_style {};
}

void WriteStdout(std::string_view text, fmt::text_style style = {}) {
	if (HasStyle(style)) {
		fmt::print(stdout, style, "{}", text);
	} else {
		std::fwrite(text.data(), 1, text.size(), stdout);
	}
	std::fflush(stdout);
}

} // namespace

namespace Log {

static bool                            g_initialized = false;
static Direction                       g_direction   = Direction::Console;
static std::filesystem::path           g_output_file;
static std::filesystem::path           g_warning_file;
static std::mutex                      g_logger_mutex;
static std::shared_ptr<spdlog::logger> g_general_logger;
static std::shared_ptr<spdlog::logger> g_warning_logger;

void Flush() {
	if (g_general_logger != nullptr) g_general_logger->flush();
	if (g_warning_logger != nullptr) g_warning_logger->flush();
}

static void SetupLogger() {
	std::lock_guard lock(g_logger_mutex);
	Flush();
	g_general_logger.reset();
	g_warning_logger.reset();

	switch (g_direction) {
		case Direction::Silent:
			g_general_logger = MakeLogger("kyty", std::make_shared<spdlog::sinks::null_sink_mt>());
			g_warning_logger =
			    MakeLogger("kyty-warnings", std::make_shared<spdlog::sinks::null_sink_mt>());
			break;
		case Direction::Console:
			g_general_logger = MakeLogger("kyty", std::make_shared<spdlog::sinks::stdout_sink_mt>());
			g_warning_logger =
			    MakeLogger("kyty-warnings", std::make_shared<spdlog::sinks::stdout_sink_mt>());
			break;
		case Direction::File:
			if (!g_output_file.empty()) {
				g_general_logger = MakeFileLogger("kyty", g_output_file);
				g_warning_logger = MakeFileLogger("kyty-warnings", g_warning_file);
			}
			break;
	}
}

static std::shared_ptr<spdlog::logger> GetLogger(Log::Level level) {
	std::lock_guard lock(g_logger_mutex);
	return level < Log::Level::Warning ? g_general_logger : g_warning_logger;
}

static void WriteImpl(Log::Level level, std::string_view text, fmt::text_style style = {}) {
	if (text.empty()) {
		return;
	}

	if (!g_initialized) {
		WriteStdout(text, style);
		return;
	}

	if (g_direction == Direction::Silent) {
		return;
	}

	if (auto logger = GetLogger(level)) {
		if (g_direction == Direction::Console && HasStyle(style)) {
			const auto styled = fmt::format(style, "{}", text);
			logger->log(ToSpdlogLevel(level), spdlog::string_view_t(styled.data(), styled.size()));
		} else {
			logger->log(ToSpdlogLevel(level), spdlog::string_view_t(text.data(), text.size()));
		}
	}
}

void WriteToConsoleAndLog(std::string_view text) {
	WriteToConsoleAndLog(Level::Info, text);
}

void WriteToConsoleAndLog(Level level, std::string_view text) {
	WriteImpl(level, text);
	if (g_initialized && g_direction != Direction::Console) {
		WriteStdout(text);
	}
	Flush();
}

void WriteWarningToConsoleAndLog(std::string_view text) {
	WriteToConsoleAndLog(Level::Warning, text);
}

void WriteFatal(std::string_view text) {
	WriteStdout(text);
	if (g_initialized && g_direction == Direction::File) WriteImpl(Level::Critical, text);
	Flush();
}

void WriteFatal(fmt::text_style style, std::string_view text) {
	WriteStdout(text, style);
	if (g_initialized && g_direction == Direction::File) {
		WriteImpl(Level::Critical, text, style);
	}
	Flush();
}

void Initialize() {
	g_initialized = true;
	switch (Config::GetPrintfDirection()) {
		case Config::LogDirection::Silent: g_direction = Direction::Silent; break;
		case Config::LogDirection::Console: g_direction = Direction::Console; break;
		case Config::LogDirection::File: g_direction = Direction::File; break;
	}
	g_output_file =
	    (g_direction == Direction::File ? Config::GetPrintfOutputFile() : std::filesystem::path {});
	g_warning_file =
	    g_output_file.empty() ? std::filesystem::path {} : WarningLogPath(g_output_file);
	SetupLogger();
}

void Shutdown() {
	Flush();
	std::lock_guard lock(g_logger_mutex);
	g_general_logger.reset();
	g_warning_logger.reset();
}

Direction GetDirection() {
	EXIT_IF(!g_initialized);
	return g_direction;
}

bool IsSilent() {
	// Before init LOGF must keep writing to stdout, so report non-silent.
	return g_initialized && g_direction == Direction::Silent;
}

void Write(std::string_view text) {
	Write(Level::Info, text);
}

void Write(fmt::text_style style, std::string_view text) {
	Write(LevelForStyle(style), style, text);
}

void Write(Level level, std::string_view text) {
	WriteImpl(level, text);
	if (level >= Level::Warning) Flush();
}

void Write(Level level, fmt::text_style style, std::string_view text) {
	WriteImpl(level, text, style);
	if (level >= Level::Warning) Flush();
}

} // namespace Log
