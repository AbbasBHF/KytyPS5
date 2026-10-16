#ifndef KYTY_COMMON_LOGGING_LOG_H_
#define KYTY_COMMON_LOGGING_LOG_H_

#include "common/common.h"

#include <fmt/color.h>
#include <fmt/printf.h>
#include <string_view>

namespace Log {

void Initialize();
void Shutdown();

struct Lifecycle {
	static constexpr const char* name               = "Log";
	static constexpr auto        initialize         = Log::Initialize;
	static constexpr auto        shutdown           = Log::Shutdown;
	static constexpr auto        emergency_shutdown = Log::Shutdown;
};

enum class Direction { Silent, Console, File };
enum class Level { Trace, Debug, Info, Warning, Error, Critical };

// Set by LoadExec for the program it starts, which continues the log file of the one it replaces
// instead of starting it over.
inline constexpr const char* RelaunchedVariable = "KYTY_RELAUNCHED";

Direction GetDirection();
bool      IsSilent();
void      Write(Level level, std::string_view text);
void      Write(Level level, fmt::text_style style, std::string_view text);
void      Write(std::string_view text);
void      Write(fmt::text_style style, std::string_view text);
void      WriteToConsoleAndLog(std::string_view text);
void      WriteToConsoleAndLog(Level level, std::string_view text);
void      WriteWarningToConsoleAndLog(std::string_view text);
void      WriteFatal(std::string_view text);
void      WriteFatal(fmt::text_style style, std::string_view text);
void      Flush();

namespace Color {

inline constexpr auto Default       = fmt::text_style {};
inline constexpr auto Red           = fmt::fg(fmt::terminal_color::red);
inline constexpr auto Green         = fmt::fg(fmt::terminal_color::green);
inline constexpr auto Yellow        = fmt::fg(fmt::terminal_color::yellow);
inline constexpr auto Magenta       = fmt::fg(fmt::terminal_color::magenta);
inline constexpr auto Cyan          = fmt::fg(fmt::terminal_color::cyan);
inline constexpr auto White         = fmt::fg(fmt::terminal_color::white);
inline constexpr auto BrightRed     = fmt::fg(fmt::terminal_color::bright_red);
inline constexpr auto BrightGreen   = fmt::fg(fmt::terminal_color::bright_green);
inline constexpr auto BrightYellow  = fmt::fg(fmt::terminal_color::bright_yellow);
inline constexpr auto BrightMagenta = fmt::fg(fmt::terminal_color::bright_magenta);
inline constexpr auto BrightWhite   = fmt::fg(fmt::terminal_color::bright_white);

} // namespace Color

} // namespace Log

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LOGF(...)                                                                                  \
	do {                                                                                           \
		if (!::Log::IsSilent()) {                                                                  \
			::Log::Write(::Log::Level::Info, ::fmt::sprintf(__VA_ARGS__));                        \
		}                                                                                          \
	} while (false)
#define LOGD(...)                                                                                  \
	do {                                                                                           \
		if (!::Log::IsSilent()) {                                                                  \
			::Log::Write(::Log::Level::Debug, ::fmt::sprintf(__VA_ARGS__));                        \
		}                                                                                          \
	} while (false)
#define LOGT(...)                                                                                  \
	do {                                                                                           \
		if (!::Log::IsSilent()) {                                                                  \
			::Log::Write(::Log::Level::Trace, ::fmt::sprintf(__VA_ARGS__));                        \
		}                                                                                          \
	} while (false)
#define LOGW(...)                                                                                  \
	do {                                                                                           \
		if (!::Log::IsSilent()) {                                                                  \
			::Log::Write(::Log::Level::Warning, ::fmt::sprintf(__VA_ARGS__));                      \
		}                                                                                          \
	} while (false)
#define LOGE(...)                                                                                  \
	do {                                                                                           \
		if (!::Log::IsSilent()) {                                                                  \
			::Log::Write(::Log::Level::Error, ::fmt::sprintf(__VA_ARGS__));                        \
		}                                                                                          \
	} while (false)
#define LOGW_COLOR(style, ...)                                                                    \
	do {                                                                                           \
		if (!::Log::IsSilent()) {                                                                  \
			::Log::Write(::Log::Level::Warning, (style), ::fmt::sprintf(__VA_ARGS__));              \
		}                                                                                          \
	} while (false)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LOGF_COLOR(style, ...)                                                                     \
	do {                                                                                           \
		if (!::Log::IsSilent()) {                                                                  \
			::Log::Write((style), ::fmt::sprintf(__VA_ARGS__));                                    \
		}                                                                                          \
	} while (false)

#endif /* KYTY_COMMON_LOGGING_LOG_H_ */
