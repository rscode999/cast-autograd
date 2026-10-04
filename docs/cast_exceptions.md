# CAST Exceptions

[Back to central documentation](Home.md)

Exceptions specific to the CAST framework.

Unless otherwise specified, all CAST exceptions directly inherit from `std::exception`.

When thrown, all CAST exceptions display the line and file where they were thrown.

## assertion_error

Thrown when a method precondition is broken.

Unlike the `cassert` assertion macro, the `assertion_error` allows a `std::string` to be passed as an error message,
allowing faulty parameters to be displayed to the user.

Thrown in the `str_assert` method.

### str_assert

*Signature:* `void str_assert(bool condition, std::string failure_message = "", std::source_location assert_location = std::source_location::current())`

If `condition` is false, throws a `cast:assertion_error` with the error message `failure_message`, recorded at the location `assert_location`.

Used over the `cassert` macro to allow for a `std::string` to be used as an error message. The `std::string` allows variable method parameters to become part of error messages, aiding in debugging.

If the NDEBUG macro is defined, this method does nothing.

**Parameters**
* `condition` (`bool`): Expression to evaluate for truth.
* `failure_message` (`std::string`): Error message if `condition` is false.
* `assert_location` (`std::source_location`): Location where the assertion is carried out- aids in debugging.

## bad_component_addition

Thrown to indicate that a network component was not added properly.

Often caused by branch IDs being out of range, or a component being added in an illegal position (i.e. a Combiner as the first component of a network).

## bad_network_config

Thrown to indicate that a Network is not in the proper configuration for an action.

The most common cause is attempting to modify an enabled Network, or attempting to train a disabled Network.

## enable_failed_error

*Subclass of `bad_network_config`.*

Thrown when the `Network::enable` method is called, but the network's configuration does not permit training and evaluation.

## not_implemented

Thrown when a method exists, but should not be used.

Thrown in the `Splitter::forward` and `Combiner::backward` methods inherited from the `NetworkComponent`'s methods.
The methods exist only to make the Splitter and Combiner into concrete classes.

## shape_error

Thrown when a component's dimensions are incompatible with a received input.

Often thrown in forward or backward passes.

## unassigned_branch_error

Thrown when a branch ID is requested from a network component, but the branch ID is not properly assigned.