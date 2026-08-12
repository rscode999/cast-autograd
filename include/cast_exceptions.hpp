#ifndef CAST_EXCEPTIONS_
#define CAST_EXCEPTIONS_

#include <source_location>
#include <stdexcept>
#include <string>


namespace cast {



    
/**
 * Thrown to indicate that a network, or a component of a network,
 * is not in the correct configuration for an action.
 * 
 * Inherits from `std::exception`
 */
class invalid_config : public std::exception {
private:
    /**
     * Error message set by the user
     */
    std::string message_;

public:
    /**
     * Creates a new exception object with the message `error_message`, thrown at the location `throw_location`.
     *
     * The error message is: `error_message` + file of `throw_location` + line number of `throw_location`
     * @param error_message message to be displayed on throw
     * @param throw_location place where the exception was thrown
     */
    explicit invalid_config(std::string error_message = "", std::source_location throw_location = std::source_location::current()) : std::exception() {
        message_ = error_message + ", file " + throw_location.file_name() + ", line " + std::to_string(throw_location.line());
    }

    /**
     * @return the exception object's error message
     */
    const char* what() const noexcept override {
        return message_.c_str();
    }
};



/**
 * Thrown to indicate that a method is incomplete, and thus should not be used.
 * 
 * Inherits from `std::logic_error`
 */
class not_implemented : public std::logic_error {

public:
    /**
     * Creates a new exception object with the message `msg`
     * @param msg error message to be displayed on throw
     */
    explicit not_implemented(std::string msg = "") : std::logic_error(msg) {
    }
};


/**
 * Thrown to indicate that a precondition is violated.
 * 
 * Inherits from `std::exception`. Enables a dynamically generated `std::string` to be used in an error message.
 */
class assertion_error : public std::exception {
private:
    /**
     * Error message set by the user
     */
    std::string full_error_message_;

public:
    /**
     * Creates a new exception object, thrown at the location `throw_location`.
     *
     * The error message is: file of `throw_location` + line number of `throw_location`
     * @param throw_location place where the exception was thrown
     */
    explicit assertion_error(std::source_location throw_location = std::source_location::current()) : std::exception() {
        full_error_message_ = "file " + std::string(throw_location.file_name()) + ", line " + std::to_string(throw_location.line());
    }

    /**
     * Creates a new exception object with the message `error_message`, thrown at the location `throw_location`.
     *
     * The error message is: `error_message` + file of `throw_location` + line number of `throw_location`
     * @param error_message message to be displayed on throw
     * @param throw_location place where the exception was thrown
     */
    explicit assertion_error(std::string error_message, std::source_location throw_location = std::source_location::current()) : std::exception() {
        full_error_message_ = error_message + ", file " + throw_location.file_name() + ", line " + std::to_string(throw_location.line()) + ", in " + throw_location.function_name();
    }

    /**
     * @return the exception object's error message
     */
    const char* what() const noexcept override {
        return full_error_message_.c_str();
    }   
};



/**
* If `condition` is false, throws a `cast::assertion_error` at the location `assert_location`.
*
* Does nothing if the `NDEBUG` macro is defined, prior to including this header (or any header that depends on this header).
* @param condition condition to check
* @param assert_location location where the assertion was enforced
*/
inline void str_assert(bool condition, std::source_location assert_location = std::source_location::current()) {
    #ifndef NDEBUG

    if(!condition) {
        throw assertion_error(assert_location);
    }

    #endif 
}



/**
* If `condition` is false, throws a `cast::assertion_error` with the message `failure_message`, at the location `assert_location`.
*
* Does nothing if the `NDEBUG` macro is defined, prior to including this header (or any header that depends on this header).
*
* Unlike standard C++ assertions, `str_assert` allows a `std::string` to be passed as an error message.
* @param condition condition to check
* @param failure_message error message to display if `condition` is false
* @param assert_location location where the assertion was enforced
*/
inline void str_assert(bool condition, std::string failure_message, std::source_location assert_location = std::source_location::current()) {
    #ifndef NDEBUG

    if(!condition) {
        throw assertion_error(failure_message, assert_location);
    }

    #endif 
}



}
#endif