#ifndef RSCODE_CUNIT_
#define RSCODE_CUNIT_

#include <cstdint>
#include <exception>
#include <iostream>
#include <source_location>
#include <stdexcept>
#include <string>
#include <unordered_map>


namespace cunit {




/**
 * Thrown if a test fails.
 * 
 * Inherits from `std::exception`
 */
class test_failed : public std::exception {
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
    explicit test_failed(std::string error_message = "", std::source_location throw_location = std::source_location::current()) : std::exception() {
        message_ = error_message + " (file " + throw_location.file_name() + ", line " + std::to_string(throw_location.line()) + ", in " + throw_location.function_name() + ")";
    }

    /**
     * @return the exception object's error message
     */
    const char* what() const noexcept override {
        return message_.c_str();
    }
};



/**
* If `expected` and `result` are not equal, prints the expected and result to `std::cerr`, 
* then throws the `cunit::test_failed` exception with the message `failure_msg`.
*
* Requires that the items being tested have an overloaded equality (`==`) and output stream insertion (`<<`) operator.
* @param expected value that a test shouhld produce
* @param result value that the test actually produced
* @param failure_msg message to print upon assertion failure, with the exception
* @param loc location where this test was carried out (for debugging purposes)
*/
template<typename T>
inline  void __assert_equals_msg(const T& expected, const T& result, std::string failure_msg = "Expected and result not equal", std::source_location loc = std::source_location::current()) {
    if(!(expected == result)) {
        std::cerr << "Expected: " << expected << "\n";
        std::cerr << "Result:   " << result << "\n";
        throw test_failed(failure_msg, loc);
    }
}



/**
* If `expected` differs from `result` by more than `tolerance`,
* throws the `cunit::test_failed` exception and flags the (0-based) index of the first mismatched element,
* printing the expected and result to `std::cerr`.
*
* Requires `expected`, `result`, and `tolerance` to be of a numeric type.
* @param expected value that a test should produce
* @param result value that the test actually produced
* @param tolerance maximum allowed difference between `expected` and `result`
* @param failure_msg message to print upon assertion failure, with the exception
* @param loc location where this test was carried out (for debugging purposes)
*/
template<typename T>
inline void __assert_almost_equals_msg_(const T& expected, const T& result, const T& tolerance, std::string failure_message = "Expected and result are not equal", std::source_location loc = std::source_location::current()) {
    if(abs(expected - result) > tolerance) {
        std::cerr << "Expected: " << expected << "\n";
        std::cerr << "Result:   " << result << "\n";
        throw test_failed(failure_message, loc);
    }
}



/**
* Equality check.
*
* If `expected` and `result` are not equal, prints the expected and result to `std::cerr`, 
* then throws the `cunit::test_failed` exception.
*
* Requires that the items being tested have an overloaded equality (`==`) and output stream insertion (`<<`) operator.
* @param expected value that a test should produce
* @param result value that the test actually produced
* @param loc location where this test was carried out (for debugging purposes)
*/
template<typename T>
inline void assert_equals(const T& expected, const T& result, std::source_location loc = std::source_location::current()) {
    __assert_equals_msg(expected, result, "Expected and result are not equal", loc);
}




/**
* Equality check (specialization for std::unordered_map<int32_t, int32_t>)
*
* If `expected` and `result` are not equal, prints each element of the expected and result to `std::cerr`, 
* then throws the `cunit::test_failed` exception.
* 
* @param expected value that a test should produce
* @param result value that the test actually produced
* @param loc location where this test was carried out (for debugging purposes)
*/
inline void assert_unordered_map_equals(const std::unordered_map<int32_t, int32_t>& expected, const std::unordered_map<int32_t, int32_t>& result, std::source_location loc = std::source_location::current()) {
    if(!(expected == result)) {
        std::cerr << "Expected: {";
        for(const auto& e : expected ) {
            std::cerr << "{" << e.first << ", " << e.second << "}, ";
        }
        std::cerr << "}\nResult:   {";
        for(auto& r : result ) {
            std::cerr << "{" << r.first << ", " << r.second << "}, ";
        }
        std::cerr << "}\n";
        throw test_failed("Expected and result are not equal", loc);
    }
}



/**
* If `expected` differs from `result` by more than `tolerance`,
* throws the `cunit::test_failed` exception and flags the (0-based) index of the first mismatched element,
* printing the expected and result to `std::cerr`.
*
* Requires `expected`, `result`, and `tolerance` to be of a numeric type.
* @param expected value that a test should produce
* @param result value that the test actually produced
* @param tolerance maximum allowed difference between `expected` and `result`
* @param loc location where this test was carried out (for debugging purposes)
*/
template<typename T>
inline void assert_almost_equals(const T& expected, const T& result, const T& tolerance, std::source_location loc = std::source_location::current()) {
    __assert_almost_equals_msg_(expected, result, tolerance, "Expected and result are not equal", loc);
}



/**
* Equality check.
*
* If `expected` and `result` are not equal, prints the expected and result to `std::cerr`, 
* then throws the `cunit::test_failed` exception.
*
* Requires that `expected` and `result` overload the equality (`==`) operator. 
* Does not require that they overload the output stream insertion (`<<`) operator.
* @param expected value that a test shouhld produce
* @param result value that the test actually produced
* @param loc location where this test was carried out (for debugging purposes)
*/
template<typename T>
inline void assert_equals_nodebug(const T& expected, const T& result, std::source_location loc = std::source_location::current()) {
    if(!(expected == result)) {
        throw test_failed("Expected and result are not equal", loc);
    }
}



/**
* Element-wise equality check.
*
* If `expected` is a different size than `result`, or any corresponding element of `expected` does not match `result`,
* throws the `cunit::test_failed` exception and flags the (0-based) index of the first mismatched element.
*
* Compares in the order of iteration only, regardless of other traits (i.e. shape in a multi-dimensional array).
*
* Requires both `expected` and `result` to have iterators.
* Each element must have an overloaded equality (`==`) and output stream insertion (`<<`) operator.
* @param expected value that a test should produce
* @param result value that the test actually produced
* @param loc location where this test was carried out (for debugging purposes)
*/
template<typename T>
inline void assert_array_equals(const T& expected, const T& result, std::source_location loc = std::source_location::current()) {
    auto expected_iterator = expected.begin();
    auto result_iterator = result.begin();

    int32_t n_elements = 0;
    for(; expected_iterator != expected.end() && result_iterator != result.end(); ++expected_iterator, ++result_iterator) {
        __assert_equals_msg(*expected_iterator, *result_iterator,
             "Expected and result are not equal at index " + std::to_string(n_elements) + " in the iteration order", loc);
        n_elements++;
    }

    if(expected_iterator == expected.end() && result_iterator != result.end()) {
        throw test_failed("Result is longer than expected", loc);
    }
    if(expected_iterator != expected.end() && result_iterator == result.end()) {
        throw test_failed("Expected is longer than result", loc);
    }
}



/**
* Element-wise equality check, with a given tolerance.
*
* If `expected` is a different size than `result`, or any corresponding element of `expected` differs from `result`'s corresponding element by more than `tolerance`,
* throws the `cunit::test_failed` exception and prints the (0-based) index of the first mismatched element to `std::cerr`.
*
* Compares in the order of iteration only, regardless of other traits (i.e. shape in a multi-dimensional array).
*
* Requires both `expected` and `result` to have iterators.
* Each element, as well as `tolerance`, must be of numeric type.
* @param expected value that a test should produce
* @param result value that the test actually produced
* @param tolerance maximum allowed difference between corresponding elements in `expected` and `result`
* @param loc location where this test was carried out (for debugging purposes)
*/
template<typename T, typename ToleranceT>
inline void assert_array_almost_equals(const T& expected, const T& result, const ToleranceT& tolerance, std::source_location loc = std::source_location::current()) {
    auto expected_iterator = expected.begin();
    auto result_iterator = result.begin();

    int32_t n_elements = 0;
    for(; expected_iterator != expected.end() && result_iterator != result.end(); ++expected_iterator, ++result_iterator) {
        const auto& current_expected = *expected_iterator;
        const auto& current_result = *result_iterator;

        __assert_almost_equals_msg_(current_expected, current_result, 
            tolerance, "Expected and result are separated by more than " + std::to_string(tolerance) + " at index " + std::to_string(n_elements) + " in the iteration order", loc);

        n_elements++;
    }

    if(expected_iterator == expected.end() && result_iterator != result.end()) {
        throw test_failed("Result is longer than expected", loc);
    }
    if(expected_iterator != expected.end() && result_iterator == result.end()) {
        throw test_failed("Expected is longer than result", loc);
    }
}



/**
* Truth test on `condition`.
*
* If `condition` is false, throws the `cunit::test_failed` exception.
* @param condition boolean value to check for truth
* @param failure_msg additional information to display if the truth test fails
* @param loc location where this test was carried out (for debugging purposes)
*/
inline void assert_true(bool condition, std::string failure_msg = "", std::source_location loc = std::source_location::current()) {
    if(!condition) {
        std::string full_failure_msg = "Truth test failed";
        full_failure_msg += (failure_msg.size() > 0) ? ("- " + failure_msg) : "";
        throw test_failed(full_failure_msg, loc);
    }
}



}
#endif 
