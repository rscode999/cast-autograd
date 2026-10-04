#include "include/cunit.hpp"
#include "../include/cast.hpp"
#include "xtensor/containers/xstorage.hpp"

#include <cassert>
#include <cstdint>
#include <immintrin.h>
#include <memory>
#include <unordered_map>
#include <xtensor/containers/xarray.hpp>

using namespace cast;
using namespace cunit;
using namespace std;
using namespace xt;

 /**
 * Tests linear 1d layer's forward operation with batches
 */
void test_linear1d_forward() {
    Linear1d l(2, 4);
    
    l.parameters() [0] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    l.parameters() [1] = {2, 2, 2, 2};

    assert_array_equals({{5, 9, 13, 17}}, l.forward({{1, 1}}));
    assert_array_equals({{5, 9, 13, 17}, {-1, -5, -9, -13}}, l.forward({{1, 1}, {-1, -1}}));
}

/**
* Tests linear 1d's forward operation with 1d inputs and 1d outputs
*/
void test_linear1d_forward_1to1() {
    Linear1d l2(1, 1);
    l2.parameters() [0] (0,0) = 2;
    l2.parameters() [1] = xt::xarray<double>{-1};

    xt::xarray<double> forward_input = xt::xarray<double>::from_shape({1, 1});
    forward_input(0,0) = 1;
    xt::xarray<double> forward_output = xt::xarray<double>::from_shape({1, 1});
    forward_output(0,0) = 1;
    assert_array_equals(forward_output, l2.forward(forward_input));

    forward_input = xt::xarray<double>::from_shape({3, 1});
    forward_input(0,0) = 2;
    forward_input(1,0) = 3;
    forward_input(2,0) = -10;
    forward_output = xt::xarray<double>::from_shape({3, 1});
    forward_output(0,0) = 3;
    forward_output(1,0) = 5;
    forward_output(2,0) = -21;
    assert_array_equals(forward_output, l2.forward(forward_input));
}



/**
* Tests Mean Squared Error and Cross Entropy loss, with and without batches
*/
void test_mse_crossentropy() {
    shared_ptr<MeanSquaredError> mse = make_shared<MeanSquaredError>();
    shared_ptr<CrossEntropy> ce = make_shared<CrossEntropy>();

    //Multi-input, no batches
    xarray<double> predicted = {0.1, 0.2, 0.3, 0.4};
    xarray<double> expected = {0.4, 0.3, 0.2, 0.1};
    assert_almost_equals(0.025, mse->compute(predicted, expected), 1e-4);
    assert_almost_equals(1.7363, ce->compute(predicted, expected), 1e-4);

    assert_array_almost_equals({-0.075, -0.025, 0.025, 0.075}, mse->compute_gradient(predicted, expected), 1e-4);
    assert_array_almost_equals({-4.0, -1.5, -0.6667, -0.25}, ce->compute_gradient(predicted, expected), 1e-4);
    assert_true(mse->compute_gradient(predicted, expected).shape() == xt::svector<size_t>{4}, "MSE gradient should have shape (4,)");
    assert_true(ce->compute_gradient(predicted, expected).shape() == xt::svector<size_t>{4}, "Cross entropy gradient should have shape (4,)");


    //Single input, no batches
    predicted = {0.4};
    expected = {1.0};
    assert_almost_equals(0.18, mse->compute(predicted, expected), 1e-3);
    assert_almost_equals(0.9163, ce->compute(predicted, expected), 1e-3);

    assert_array_almost_equals({-0.6}, mse->compute_gradient(predicted, expected), 1e-4);
    assert_array_almost_equals({-2.5}, ce->compute_gradient(predicted, expected), 1e-4);
    assert_true(mse->compute_gradient(predicted, expected).shape() == xt::svector<size_t> {1}, "MSE loss gradient should have shape (1,)");
    assert_true(ce->compute_gradient(predicted, expected).shape() == xt::svector<size_t> {1}, "Cross entropy loss gradient should have shape (1,)");


    //Multi-input with multiple dimensions, no batches
    predicted = {{0.1, 0.1, 0.2}, {0.2, 0.1, 0.3}};
    expected = {{0.1, 0.1, 0.2}, {0.2, 0.3, 0.1}};
    assert_almost_equals(0.006667, mse->compute(predicted, expected), 1e-4);
    assert_almost_equals(1.9155, ce->compute(predicted, expected), 1e-4);

    assert_array_almost_equals({{0.0, 0.0, 0.0}, {0.0, -0.0333, 0.0333}}, mse->compute_gradient(predicted, expected), 1e-4);
    assert_array_almost_equals({{-1.0, -1.0, -1.0}, {-1.0, -3.0, -0.3333}}, ce->compute_gradient(predicted, expected), 1e-4);
    assert_true(mse->compute_gradient(predicted, expected).shape() == xt::svector<size_t>{2, 3}, "MSE loss gradient should have shape (2,3)");
    assert_true(ce->compute_gradient(predicted, expected).shape() == xt::svector<size_t>{2, 3}, "Cross entropy loss gradient should have shape (2,3)");


    //Multi-input over batches
    mse->set_batch_size(2);
    ce->set_batch_size(2);
    predicted = {{0.1, 0.1, 0.8}, {0.2, 0.1, 0.7}};
    expected = {{0.5, 0.1, 0.4}, {0.2, 0.3, 0.5}};
    assert_almost_equals(0.06667, mse->compute(predicted, expected), 1e-4);
    assert_almost_equals(1.3309, ce->compute(predicted, expected), 1e-4);

    assert_array_almost_equals({{-0.1333, 0.0, 0.13333}, {0.0, -0.0667, 0.0667}}, mse->compute_gradient(predicted, expected), 1e-4);
    assert_array_almost_equals({{-2.5, -0.5, -0.25}, {-0.5, -1.5, -0.3571}}, ce->compute_gradient(predicted, expected), 1e-4);
    assert_true(mse->compute_gradient(predicted, expected).shape() == xt::svector<size_t>{2, 3}, "MSE loss gradient should have shape (2,3)");
    assert_true(ce->compute_gradient(predicted, expected).shape() == xt::svector<size_t>{2, 3}, "Cross entropy loss gradient should have shape (2,3)");


    //multiple inputs in a batch
    predicted = {{{0.1, 0.2}, {0.3, 0.4}}, {{0.1, 0.2}, {0.3, 0.4}}};
    expected = {{{0.1, 0.2}, {0.3, 0.4}}, {{0.1, 0.2}, {0.3, 0.4}}};
    assert_almost_equals(0., mse->compute(predicted, expected), 1e-4);
    assert_almost_equals(1.27985, ce->compute(predicted, expected), 1e-4);

    assert_array_almost_equals(xt::zeros_like(predicted), mse->compute_gradient(predicted, expected), 1e-4);
    assert_array_almost_equals(xt::xarray<double>({2,2,2}, -0.5), ce->compute_gradient(predicted, expected), 1e-4);
    assert_true(mse->compute_gradient(predicted, expected).shape() == xt::svector<size_t>{2, 2, 2}, "MSE loss gradient should have shape (2,2,2)");
    assert_true(ce->compute_gradient(predicted, expected).shape() == xt::svector<size_t>{2, 2, 2}, "Cross entropy loss gradient should have shape (2,2,2)");


    //Batch size 1
    mse->set_batch_size(1);
    ce->set_batch_size(1);
    predicted = {{0.3, 0.4, 0.4}};
    expected = {{0.1, 0.1, 0.8}};
    assert_almost_equals(0.04835, mse->compute(predicted, expected), 1e-4);
    assert_almost_equals(0.9451, ce->compute(predicted, expected), 1e-4);

    assert_array_almost_equals({{0.06667, 0.1, -0.13333}}, mse->compute_gradient(predicted, expected), 1e-4);
    assert_array_almost_equals({{-0.3333, -0.25, -2.0}}, ce->compute_gradient(predicted, expected), 1e-4);
    assert_true(mse->compute_gradient(predicted, expected).shape() == xt::svector<size_t>{1, 3}, "MSE loss gradient should have shape (1,3)");
    assert_true(ce->compute_gradient(predicted, expected).shape() == xt::svector<size_t>{1, 3}, "Cross entropy loss gradient should have shape (1,3)");

}



/**
* Trains on the XOR dataset. Does not train in batches.
*/
void test_train_no_batch() {
Network net = Network();

    //(2,4) -> (4,1)
    net.add_operator(make_shared<Linear1d>(2, 4));
    net.add_operator(make_shared<Sigmoid>());
    net.add_operator(make_shared<Linear1d>(4, 1));

    shared_ptr<MeanSquaredError> loss_calc = make_shared<MeanSquaredError>();
    net.set_loss_calculator(loss_calc);
    net.set_optimizer(make_shared<SGD>(0.02, 0.9));
    
    net.enable();

    vector<xarray<double>> inputs = {
        xarray<double>{0, 0},
        xarray<double>{0, 1},
        xarray<double>{1, 0},
        xarray<double>{1, 1}
    };

    vector<xarray<double>> expected_outputs;
    expected_outputs.push_back(xt::xarray<double>({1}, 0.0));
    expected_outputs.push_back(xt::xarray<double>({1}, 1.0));
    expected_outputs.push_back(xt::xarray<double>({1}, 1.0));
    expected_outputs.push_back(xt::xarray<double>({1}, 0.0));


    //Train
    for(int e = 0; e < 1000; e++) {
        double loss = 0;
        for(int i = 0; i < (int)inputs.size(); i++) {
            xt::xarray<double> prediction = net.forward(inputs[i]);
            loss += loss_calc->compute(prediction, expected_outputs[i]);
            net.backward(prediction, expected_outputs[i]);
            net.optimize();
        }
    }


    //Get total loss and final predictions
    double loss = 0;
    for(int i = 0; i < (int)inputs.size(); i++) {
        xt::xarray<double> prediction = net.forward(inputs[i]);
        loss += loss_calc->compute(prediction, expected_outputs[i]);
        assert_array_almost_equals(expected_outputs[i], prediction, 0.05);
    }
    assert_true(loss < 1e-3, "Loss must be below 1e-3 (got " + std::to_string(loss) + ")");
}



/**
* Trains with a batch size of 1 on the XOR dataset
*/
void test_train_batch_1() {
    Network net = Network();

    //(2,4) -> (4,1)
    net.add_operator(make_shared<Linear1d>(2, 4));
    net.add_operator(make_shared<Sigmoid>());
    net.add_operator(make_shared<Linear1d>(4, 1));

    shared_ptr<MeanSquaredError> loss_calc = make_shared<MeanSquaredError>();
    net.set_loss_calculator(loss_calc);
    net.set_optimizer(make_shared<SGD>(0.02, 0.9));
    
    net.set_batch_size(1);
    net.enable();

    vector<xarray<double>> inputs = {
        xarray<double>{{0, 0}},
        xarray<double>{{0, 1}},
        xarray<double>{{1, 0}},
        xarray<double>{{1, 1}}
    };

    vector<xarray<double>> expected_outputs;
    expected_outputs.push_back(xt::xarray<double>({1, 1}, 0.0));
    expected_outputs.push_back(xt::xarray<double>({1, 1}, 1.0));
    expected_outputs.push_back(xt::xarray<double>({1, 1}, 1.0));
    expected_outputs.push_back(xt::xarray<double>({1, 1}, 0.0));


    //Train
    for(int e = 0; e < 1000; e++) {
        double loss = 0;
        for(int i = 0; i < (int)inputs.size(); i++) {
            xt::xarray<double> prediction = net.forward(inputs[i]);
            loss += loss_calc->compute(prediction, expected_outputs[i]);
            net.backward(prediction, expected_outputs[i]);
            net.optimize();
        }
    }

    //Get total loss
    double loss = 0;
    for(int i = 0; i < (int)inputs.size(); i++) {
        xt::xarray<double> prediction = net.forward(inputs[i]);
        loss += loss_calc->compute(prediction, expected_outputs[i]);
        assert_array_almost_equals(expected_outputs[i], prediction, 0.05);
    }
    assert_true(loss < 1e-3, "Loss must be below 1e-3 (got " + std::to_string(loss) + ")");
}


/**
* Trains with a batch size of 2 on the XOR dataset
*/
void test_train_batch_2() {
    Network net = Network();

    //(2,4) -> (4,1)
    net.add_operator(make_shared<Linear1d>(2, 4));
    net.add_operator(make_shared<Sigmoid>());
    net.add_operator(make_shared<Linear1d>(4, 1));

    shared_ptr<MeanSquaredError> loss_calc = make_shared<MeanSquaredError>();
    net.set_loss_calculator(loss_calc);
    net.set_optimizer(make_shared<SGD>(0.02, 0.9));
    
    net.set_batch_size(2);
    net.enable();

    vector<xarray<double>> inputs = {
        xarray<double>{{0, 0}, {0,1}},
        xarray<double>{{1, 0}, {1, 1}},
    };

    vector<xarray<double>> expected_outputs;
    xarray<double> first = {0.0, 1.0};
    expected_outputs.push_back(first.reshape({2, 1}));
    xarray<double> second = {1.0, 0.0};
    expected_outputs.push_back(second.reshape({2, 1}));

    //Train
    for(int e = 0; e < 1000; e++) {
        double loss = 0;
        for(int i = 0; i < (int)inputs.size(); i++) {
            xt::xarray<double> prediction = net.forward(inputs[i]);
            loss += loss_calc->compute(prediction, expected_outputs[i]);
            net.backward(prediction, expected_outputs[i]);
            net.optimize();
        }
    }

    //Get total loss
    double loss = 0;
    for(int i = 0; i < (int)inputs.size(); i++) {
        xt::xarray<double> prediction = net.forward(inputs[i]);
        loss += loss_calc->compute(prediction, expected_outputs[i]);
        assert_array_almost_equals(expected_outputs[i], prediction, 0.05);
    }
    assert_true(loss < 1e-3, "Loss must be below 1e-3 (got " + std::to_string(loss) + ")");
}


/**
* Builds a classifier with multiple branches, then verifies the branching structure.
*/
void test_create_branch() {
    Network net;
    std::shared_ptr<MeanSquaredError> loss_calc = make_shared<MeanSquaredError>();
    net.set_loss_calculator(loss_calc);
    net.set_optimizer(make_shared<SGD>(0.02, 0.9));

    /*
    This is a very complicated XOR classifier.
    l1 2-4 > branch (1)  0 > l1 4-1                                                 > combiner (9)  
                         1 > branch (3)  1 > l1 4-1 (4) > sigmoid (6) > combiner (8) ^    
                                         2 > l1 4-1 (5) > sigmoid (7) ^

    Upon addition of the combiners, the test ensures that the combiners have been properly added.
    */

    net.add_operator(make_shared<Linear1d>(2, 4));

    net.add_splitter(2);
    net.add_operator(make_shared<Linear1d>(4, 1));
    net.add_splitter(2, 1);
    net.add_operator(make_shared<Linear1d>(4, 1), 1);
    
    net.add_operator(make_shared<Linear1d>(4, 1), 2);
    net.add_operator(make_shared<Sigmoid>(), 1);
    net.add_operator(make_shared<Sigmoid>(), 2);

    ////////////////////////////////////////////////////////////////////////////////////

    //At this point, no combiners have been added. The available branch IDs should be 0, 1, 2
    std::unordered_map<int32_t, int32_t> expected_branch_ids = {
        {0, 2}, //Branch 0 ends at components index 2 (3rd component added)
        {1, 6}, //Branch 1 ends at components index 6 (7th component added)
        {2, 7} //Branch 2 ends at components index 7
    };
    assert_unordered_map_equals(expected_branch_ids, net.active_branch_id_heads());


    //Try to add a combiner to its own branch, and verify that it throws an exception
    try {
        net.add_combiner({2}, 2);
        throw test_failed("Merging branch 2 from within branch 2 should cause a cast::bad_component_addition");
    }
    catch(bad_component_addition& e) {
        //Test passed
    }

    //Try to merge a branch that does not exist
    try {
        net.add_combiner({3}, 2);
        throw test_failed("Merging branch 3, which does not exist, should cause a cast::bad_component_addition");
    }
    catch(bad_component_addition& e) {
        //Test passed
    }

    net.add_combiner({2}, 1);
    //After merging branch 2 into branch 1, there should be 2 remaining branch IDs (0 and 1)
    expected_branch_ids = {
        {0, 2}, //Branch 0 ends at components index 2 (3rd component added)
        {1, 8}, //Branch 1 ends at components index 8 (9th component added)
    };
    assert_unordered_map_equals(expected_branch_ids, net.active_branch_id_heads());


    net.add_combiner({1}, 0);
  
    net.enable();

    ////////////////////////////////////////////////////////////////////////////////////

    //Check predecessors and successors

    //SIGMOID (the first one added)
    //Should have predecessor in branch 1, index 4
    std::shared_ptr<NetworkComponent> comp6 = net.component_at(6);
    assert_unordered_map_equals(
        std::unordered_map<int32_t, int32_t> {{1, 4}},
        comp6->predecessors()
    );
    //Successor in branch 1, index 8 (the combiner)
    assert_unordered_map_equals(
        std::unordered_map<int32_t, int32_t> {{1, 8}},
        comp6->successors()
    );

    //FIRST COMBINER
    //Should have predecessor in branches 1 and 2
    std::shared_ptr<NetworkComponent> comp8 = net.component_at(8); //The first combiner
    assert_unordered_map_equals(
        std::unordered_map<int32_t, int32_t> {{1, 6}, {2, 7}},
        comp8->predecessors()
    );
    //Should have successor in branch 0
    assert_unordered_map_equals(
        std::unordered_map<int32_t, int32_t> {{0, 9}},
        comp8->successors()
    );

    //LAST COMBINER
    //Should have predecessor in branches 0 and 2
    std::shared_ptr<NetworkComponent> comp9 = net.component_at(9); //The last combiner
        assert_unordered_map_equals(
        std::unordered_map<int32_t, int32_t> {{0, 2}, {1, 8}},
        comp9->predecessors()
    );
    //Should have no successor
    assert_unordered_map_equals(
        std::unordered_map<int32_t, int32_t>(),
        comp9->successors()
    );
}



/**
* Creates a network with a splitter first
*/
void test_create_branch_splitter_first() {
    Network net;
    net.add_splitter(3);

    //Make sure the network has registered new leaf nodes
    assert_unordered_map_equals({{0,0}, {1,0}, {2,0}}, net.active_branch_id_heads());

    net.add_operator(make_shared<Linear1d>(2, 4), 0);
    net.add_operator(make_shared<Linear1d>(2, 4), 1);
    net.add_operator(make_shared<Linear1d>(2, 4), 2);

    //Make sure the splitter has the proper predecessors and successors
    assert_unordered_map_equals({{0,1}, {1,2}, {2,3}}, net.component_at(0)->successors());
    assert_unordered_map_equals(unordered_map<int32_t, int32_t>(), net.component_at(0)->predecessors());

    //Make sure the new components have the proper predecessors and successors
    assert_unordered_map_equals({{0,0}}, net.component_at(1)->predecessors());
    assert_unordered_map_equals({{0,0}}, net.component_at(2)->predecessors());
    assert_unordered_map_equals({{0,0}}, net.component_at(3)->predecessors());
    assert_unordered_map_equals(unordered_map<int32_t, int32_t>(), net.component_at(1)->successors());
    assert_unordered_map_equals(unordered_map<int32_t, int32_t>(), net.component_at(2)->successors());
    assert_unordered_map_equals(unordered_map<int32_t, int32_t>(), net.component_at(3)->successors());

    //Combine into branch 2
    net.add_combiner({1,0}, 2); //Component ID = 4

    //Check network branch heads
    assert_unordered_map_equals({{2, 4}}, net.active_branch_id_heads());

    //Combiner has proper predecessors and successors
    assert_unordered_map_equals({{0,1}, {1,2}, {2,3}}, net.component_at(4)->predecessors());
    assert_unordered_map_equals(unordered_map<int32_t, int32_t>(), net.component_at(4)->successors());

    //Operators have proper predecessors and successors
    assert_unordered_map_equals({{0,0}}, net.component_at(1)->predecessors());
    assert_unordered_map_equals({{0,0}}, net.component_at(2)->predecessors());
    assert_unordered_map_equals({{0,0}}, net.component_at(3)->predecessors());
    assert_unordered_map_equals({{2,4}}, net.component_at(1)->successors());
    assert_unordered_map_equals({{2,4}}, net.component_at(2)->successors());
    assert_unordered_map_equals({{2,4}}, net.component_at(3)->successors());
}



/**
* Tests the very weird case where a network consists only of control flow components
*/
void test_create_control_flow_only() {
    Network net;
    net.add_splitter(2);
    net.add_splitter(2);

    //There should be 3 branches in total
    assert_unordered_map_equals({{0,1}, {1,0}, {2,1}}, net.active_branch_id_heads());

    //Splitter 0 has a successor: branch 0, ID 1
    assert_unordered_map_equals({{0,1}}, net.component_at(0)->successors());
    //Splitter 1 should have no successors and one predecessor
    assert_unordered_map_equals(unordered_map<int32_t, int32_t>(), net.component_at(1)->successors());
    assert_unordered_map_equals({{0, 0}}, net.component_at(1)->predecessors());

    //Splitter 0 should have no predecessors
    assert_unordered_map_equals(unordered_map<int32_t, int32_t>(), net.component_at(0)->predecessors());
    //Splitter 1 should have one predecessor: branch 0, component 0
    assert_unordered_map_equals({{0, 0}}, net.component_at(1)->predecessors());

    net.add_combiner({2}, 1); //Component ID: 2

    //Network should have 2 available branches now
    assert_unordered_map_equals({{0, 1},{1, 2}}, net.active_branch_id_heads());
    //Branch 0's ending component is component 1. Branch 1's ending component is component 2

    assert_unordered_map_equals({{0,1}}, net.component_at(2)->predecessors());
    //Splitter 1 should have a successor now
    assert_unordered_map_equals({{1,2}}, net.component_at(1)->successors());

    net.add_combiner({0}, 1); //Component ID: 3

    //Network has only one active branch, 1, with component 3 as its head
    assert_unordered_map_equals({{1, 3}}, net.active_branch_id_heads());

    //New combiner's predecessors are: branch 0, component 1; branch 1, component 2
    assert_unordered_map_equals({{0,1}, {1,2}}, net.component_at(3)->predecessors());
    assert_unordered_map_equals(unordered_map<int32_t, int32_t>(), net.component_at(3)->successors());

    //Splitter 1's successor is combiner 3. Original predecessors are kept
    assert_unordered_map_equals({{1, 3}}, net.component_at(1)->successors());
    assert_unordered_map_equals({{0, 0}}, net.component_at(1)->predecessors());
    //Combiner 2's successor is combiner 3. Original predecessors are kept
    assert_unordered_map_equals({{1, 3}}, net.component_at(2)->successors());
    assert_unordered_map_equals({{0, 1}, {0, 0}}, net.component_at(2)->predecessors());
}



/**
* Tests the forward and backward methods of a splitter.
*/
void test_splitter_forward_backward() {
    //Backward: Should duplicate its inputs
    shared_ptr<Splitter> s1 = make_shared<Splitter>(2);
    shared_ptr<Splitter> s2 = make_shared<Splitter>(3);
    xarray<double> in = {1, 2, 3};

    vector<xarray<double>> out1 = s1->forward(in, true);
    vector<xarray<double>> out2 = s2->forward(in, true);
    assert_equals((size_t)2, out1.size());
    for(xarray<double> o : out1) {
        assert_array_equals(in, o);
    }
    assert_equals((size_t)3, out2.size());
    for(xarray<double> o : out2) {
        assert_array_equals(in, o);
    }


    shared_ptr<Splitter> s3 = make_shared<Splitter>(2);
    shared_ptr<Splitter> s4 = make_shared<Splitter>(3);

    //Backward: should give output after 3 inputs
    vector<xarray<double>> inputs = {
        xarray<double>{0, 1, 2, 3}, //1
        xarray<double>{-1, -1, -1, -1}, //2
        xarray<double>{0, 1, 2, 3}, //3
        xarray<double>{5, 5, 5, 0} //4
    };

    for (int i = 1; i <= inputs.size(); i++) {
        xarray<double> out3 = s3->backward(inputs[i-1]);
        xarray<double> out4 = s4->backward(inputs[i-1]);

        //on odd-numbered inputs, s3 should produce an empty output
        if(i % 2 == 1) {
            assert_true(out3.size() == 0, "Splitter to 2 branches should produce empty outputs on odd-numbered inputs");
        }
        //input 2 should be the sum of inputs 1 and 2
        else if(i == 2) {
            assert_equals(xarray<double> {-1, 0, 1, 2}, out3);
        }
        //input 4 should be the sum of inputs 3 and 4
        else if(i == 4) {
            assert_equals(xarray<double> {5, 6, 7, 3}, out3);
        }

        //if on input 3, c2 should produce an output. otherwise, should be empty
        if(i == 3) {
            assert_array_equals(xarray<double> {-1, 1, 3, 5}, out4);
        }
        else {
            assert_true(out4.size() == 0, "Splitter to 3 other branches should produce empty outputs except on output 3");
        }
    }
}



void test_combiner_forward_backward() {
    shared_ptr<Combiner> c1 = make_shared<Combiner>(initializer_list<int32_t>{1});
    shared_ptr<Combiner> c2 = make_shared<Combiner>(initializer_list<int32_t>{1, 2});

    //Forward: should give output after 2 (c1) or 3 (c2) inputs
    vector<xarray<double>> inputs = {
        xarray<double>{0, 1, 2, 3}, //1
        xarray<double>{-1, -1, -1, -1}, //2
        xarray<double>{0, 1, 2, 3}, //3
        xarray<double>{5, 5, 5, 0} //4
    };

    for (int i = 1; i <= inputs.size(); i++) {
        xarray<double> out1 = c1->forward(inputs[i-1]);
        xarray<double> out2 = c2->forward(inputs[i-1]);

        //on odd-numbered inputs, c1 should produce an empty output
        if(i % 2 == 1) {
            assert_true(out1.size() == 0, "Combiner merging 1 other branch should produce empty outputs on odd-numbered inputs");
        }
        //input 2 should be the sum of inputs 1 and 2
        else if(i == 2) {
            assert_equals(xarray<double> {-1, 0, 1, 2}, out1);
        }
        //input 4 should be the sum of inputs 3 and 4
        else if(i == 4) {
            assert_equals(xarray<double> {5, 6, 7, 3}, out1);
        }

        //if on input 3, c2 should produce an output. otherwise, should be empty
        if(i == 3) {
            assert_array_equals(xarray<double> {-1, 1, 3, 5}, out2);
        }
        else {
            assert_true(out2.size() == 0, "Combiner merging 2 other branches should produce empty outputs except on output 3");
        }
    }


    //Backward: Should duplicate its inputs
    shared_ptr<Combiner> c3 = make_shared<Combiner>(initializer_list<int32_t>{1});
    shared_ptr<Combiner> c4 = make_shared<Combiner>(initializer_list<int32_t>{1, 10});
    xarray<double> in = {1, 2, 3};

    vector<xarray<double>> out3 = c3->backward(in, true);
    vector<xarray<double>> out4 = c4->backward(in, true);
    assert_equals((size_t)2, out3.size());
    for(xarray<double> o : out3) {
        assert_array_equals(in, o);
    }
    assert_equals((size_t)3, out4.size());
    for(xarray<double> o : out3) {
        assert_array_equals(in, o);
    }
}   



/**
* Builds a classifier involving multiple branches, trains it on the XOR dataset, and checks that the classifier converged.
*/
void test_train_branch() {

    Network net;
    std::shared_ptr<MeanSquaredError> loss_calc = make_shared<MeanSquaredError>();
    net.set_loss_calculator(loss_calc);
    net.set_optimizer(make_shared<SGD>(0.02, 0.9));

    /*
    This is a very complicated XOR classifier.
    l1 2-4 > branch (1)  0 > l1 4-1                                                 > combiner (9)  
                         1 > branch (3)  1 > l1 4-1 (4) > sigmoid (6) > combiner (8) ^    
                                         2 > l1 4-1 (5) > sigmoid (7) ^
    */

    net.add_operator(make_shared<Linear1d>(2, 4));

    net.add_splitter(2);
    net.add_operator(make_shared<Linear1d>(4, 1));
    net.add_splitter(2, 1);
    net.add_operator(make_shared<Linear1d>(4, 1), 1);
    
    net.add_operator(make_shared<Linear1d>(4, 1), 2);
    net.add_operator(make_shared<Sigmoid>(), 1);
    net.add_operator(make_shared<Sigmoid>(), 2);
    net.add_combiner({2}, 1);

    net.add_combiner({1}, 0);
  
    net.enable();
   
    vector<xarray<double>> inputs = {
        xarray<double>{0, 0},
        xarray<double>{0, 1},
        xarray<double>{1, 0},
        xarray<double>{1, 1}
    };

    vector<xarray<double>> expected_outputs = {
        xarray<double>{0},
        xarray<double>{1},
        xarray<double>{1},
        xarray<double>{0}
    };

    
    for(int e = 0; e < 1000; e++) {
        for(int i = 0; i < (int)inputs.size(); i++) {
            xt::xarray<double> prediction = net.forward(inputs[i]);
            net.backward(prediction, expected_outputs[i]);
            net.optimize();
        }
    }


    for(int i = 0; i < (int)inputs.size(); i++) {
        xarray<double> prediction = net.forward(inputs[i]);

        //Check that the prediction is of the proper shape
        assert_true(prediction.shape() == expected_outputs[i].shape(), "Prediction and expected shapes not equal (expected outputs index " + std::to_string(i) + ")");
        //Each prediction element is within 0.05 of the expected output
        assert_array_almost_equals(expected_outputs[i], prediction, 0.05);
  }
}



/**
* Trains on the binary to one-hot dataset.
* Uses cross-entropy and softmax, while using the copy constructor to make another network with a nonzero batch size.
*/
void test_ce_sm_batch_assign() {

    const int32_t N_INPUTS = 4;
    const int32_t N_OUTPUTS = pow(2, N_INPUTS);
    const int32_t N_EPOCHS = 2000;

    //Make the dataset
    vector<xarray<double>> inputs;
    vector<xarray<double>> expected_outputs;

    for (int i = 0; i < N_OUTPUTS; i++) {
        xarray<double> input = xt::zeros<double>({N_INPUTS});
        xarray<double> output = xt::zeros<double>({N_OUTPUTS});

        //Inputs to binary value
        for (int j = 0; j < N_INPUTS; j++) {
            input(j) = (i >> j) & 1;
        }

        //Output[i] to 1
        output(i) = 1.0;

        inputs.emplace_back(input);
        expected_outputs.emplace_back(output);
    }


    //Make the batch training dataset, with 1/2 of the inputs and expected outputs per index
    vector<xt::xarray<double>> input_batches(2);
    vector<xt::xarray<double>> expected_outputs_batches(2);
    const size_t half = N_OUTPUTS / 2;

    for (int batch = 0; batch < 2; ++batch) {
        input_batches[batch] = xt::zeros<double>({half, static_cast<size_t>(N_INPUTS)});

        expected_outputs_batches[batch] = xt::zeros<double>({half, static_cast<size_t>(N_OUTPUTS)});

        for (size_t i = 0; i < half; ++i) {
            for (size_t j = 0; j < N_INPUTS; ++j) {
                input_batches[batch](i, j) =
                    inputs[batch * half + i](j);
            }

            for (size_t j = 0; j < N_OUTPUTS; ++j) {
                expected_outputs_batches[batch](i, j) =
                    expected_outputs[batch * half + i](j);
            }
        }
    }

    Network net;
    net.add_splitter(3);

    net.add_operator(make_shared<Linear1d>(N_INPUTS, N_OUTPUTS/2), 0);
    net.add_operator(make_shared<Linear1d>(N_INPUTS, N_OUTPUTS/2), 1);
    net.add_operator(make_shared<Linear1d>(N_INPUTS, N_OUTPUTS/2), 2);
    
    net.add_combiner({1,0}, 2);

    net.add_operator(make_shared<Linear1d>(N_OUTPUTS/2, N_OUTPUTS), 2);
    net.add_operator(make_shared<Sigmoid>(), 2);
    net.add_operator(make_shared<Linear1d>(N_OUTPUTS, N_OUTPUTS), 2);
    net.add_operator(make_shared<Sigmoid>(), 2);
    net.add_operator(make_shared<Linear1d>(N_OUTPUTS, N_OUTPUTS), 2);
    net.add_operator(make_shared<Softmax>(1), 2);

    std::shared_ptr<CrossEntropy> ce_loss = make_shared<CrossEntropy>();
    net.set_loss_calculator(ce_loss);
    net.set_optimizer(make_shared<SGD>(0.005, 0.9));

    net.enable();

    Network net2 = net;
    net2.set_batch_size(N_OUTPUTS / 2);
    net2.enable();

    //Train the networks
    for (int32_t e = 0; e < N_EPOCHS; e++) {
        double loss = 0;
        ce_loss->set_batch_size(0);
        for (int32_t i = 0; i < N_OUTPUTS; i++) {
            xarray<double> out = net.forward(inputs[i]);
            loss += ce_loss->compute(out, expected_outputs[i]);
            net.backward(out, expected_outputs[i]);
            net.optimize();
        }

        // if(e % 200 == 0) {
        //     cout << e << " completed, loss " << loss << endl;
        // }

        loss = 0;
        ce_loss->set_batch_size(N_OUTPUTS / 2);
        for(int32_t i = 0; i < 2; i++) {
            xarray<double> out_batches = net2.forward(input_batches[i]);
            loss += ce_loss->compute(out_batches, expected_outputs_batches[i]);
            net2.backward(out_batches, expected_outputs_batches[i]);
            net2.optimize();
        }

        // if(e % 200 == 0) {
        //     cout << e << " completed, loss " << loss << endl;
        // }
    }

    //Check that the element-wise difference is less than 0.5 from its expected
    for(int32_t i = 0; i < N_OUTPUTS; i++) {
        xarray<double> out = net.forward(inputs[i]);
        // cout << out << endl;
        assert_array_almost_equals(expected_outputs[i], out, 0.5);
    }
    for(int32_t i = 0; i < 2; i++) {
        xarray<double> out = net2.forward(input_batches[i]);
        // cout << out << endl;
        assert_array_almost_equals(expected_outputs_batches[i], out, 0.5);
    }
}




int main() {
    test_linear1d_forward();
    test_linear1d_forward_1to1();
    test_mse_crossentropy();
    test_train_no_batch();
    test_train_batch_1();
    test_train_batch_2();
    test_create_branch();
    test_create_branch_splitter_first();
    test_create_control_flow_only();
    test_splitter_forward_backward();
    test_combiner_forward_backward();
    test_train_branch();
    test_ce_sm_batch_assign();
    cout << "Tests passed" << endl;
}