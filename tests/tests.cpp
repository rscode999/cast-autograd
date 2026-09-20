#include "include/cunit.hpp"
#include "../include/cast.hpp"

#include <cassert>
#include <cstdint>
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
    
    net.enable();

    net.set_batch_size(1);

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
    
    net.enable();

    net.set_batch_size(2);

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






int main() {
    test_linear1d_forward();
    test_linear1d_forward_1to1();
    test_train_no_batch();
    test_train_batch_1();
    test_train_batch_2();
    test_create_branch();
    test_splitter_forward_backward();
    test_combiner_forward_backward();
    test_train_branch();
    cout << "Tests passed" << endl;
}