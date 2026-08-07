#ifndef CAST_EXCEPTIONS_
#define CAST_EXCEPTIONS_

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
    std::string message;

public:
    /**
     * Creates a new exception object with the message `msg`
     * @param msg error message to be displayed on throw
     */
    explicit invalid_config(std::string msg = "") : std::exception() {
        message = msg;
    }

    /**
     * @return the exception object's error message
     */
    const char* what() const noexcept override {
        return message.c_str();
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



}
#endif