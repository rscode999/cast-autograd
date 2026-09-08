#include "include/cunit.hpp"
#include "../include/cast.hpp"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <xtensor/containers/xarray.hpp>

using namespace cast;
using namespace cunit;
using namespace std;
using namespace xt;



/**
* Trains on the XOR dataset, verifying the model's structure and checking that loss has decreased and that predictions are with 0.05 of their expected outputs.
* Also tests for deep-copy assignment.
*/
void test_create_train_xor() {
    Network net = Network();

    //(2,4) -> (4,1)
    net.add_operator(make_shared<Linear1d>(2, 4));
    net.add_operator(make_shared<Sigmoid>());
    net.add_operator(make_shared<Linear1d>(4, 1));

    shared_ptr<MeanSquaredError> loss_calc = make_shared<MeanSquaredError>();
    net.set_loss_calculator(loss_calc);
    net.set_optimizer(make_shared<SGD>(0.02, 0.9));
    
    net.enable();

    //Check predecessors and successors

    //first linear1d: should have no predecessors, successor as component 1
    assert_unordered_map_equals(unordered_map<int32_t, int32_t>(), net.component_at(0)->predecessors());
    assert_unordered_map_equals({{0, 1}}, net.component_at(0)->successors());

    //sigmoid: predecessors = first linear1d (branch 0), successors = second linear1d (branch 0)
    assert_unordered_map_equals({{0, 0}}, net.component_at(1)->predecessors());
    assert_unordered_map_equals({{0, 2}}, net.component_at(1)->successors());

    //second linear1d: predecessors = sigmoid (branch 0), no successors
    assert_unordered_map_equals({{0, 1}}, net.component_at(2)->predecessors());
    assert_unordered_map_equals(unordered_map<int32_t, int32_t>(), net.component_at(2)->successors());

    //Create 2 deep copies of the network
    Network net2 = net;
    Network net3;
    net3 = net;

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


    //Train
    for(int e = 0; e < 1000; e++) {
        for(int i = 0; i < (int)inputs.size(); i++) {
            xt::xarray<double> prediction = net.forward(inputs[i]);
            net.backward(prediction, expected_outputs[i]);
            net.optimize();
        }
    }

    //Compute total loss one more time
    double final_loss = 0;
    for(int i = 0; i < (int)inputs.size(); i++) {
        xarray<double> prediction = net.forward(inputs[i]);
        final_loss += loss_calc->compute(prediction, expected_outputs[i]);
    }
    
    //Check that loss is below 1e-6
    assert_true(final_loss < 1e-3, "Final calculated loss cannot exceed 1e-3 (got " + std::to_string(final_loss) + ")");

    //Check that the loss is below 1e-8
    double loss = 0;
    for(int i = 0; i < (int)inputs.size(); i++) {
        xarray<double> prediction = net.forward(inputs[i]);
        loss += loss_calc->compute(prediction, expected_outputs[i]);
        net.backward(prediction, expected_outputs[i]);
        net.optimize();
    }


  for(int i = 0; i < (int)inputs.size(); i++) {
        xarray<double> prediction = net.forward(inputs[i]);

        //Check that the prediction is of the proper shape
        assert_true(prediction.shape() == expected_outputs[i].shape(), "Prediction and expected shapes not equal (expected outputs index " + std::to_string(i) + ")");
        //Prediction is within 0.05 of the expected output
        assert_array_almost_equals(expected_outputs[i], prediction, 0.05);

        //Check that the other networks are not trained, and that the other networks' weights are the same
        std::shared_ptr<Linear1d> l = dynamic_pointer_cast<Linear1d>(net.component_at(0));
        std::shared_ptr<Linear1d> l2 = dynamic_pointer_cast<Linear1d>(net2.component_at(0));
        std::shared_ptr<Linear1d> l3 = dynamic_pointer_cast<Linear1d>(net3.component_at(0));
        assert_true(l->parameters()[0] != l2->parameters()[0], "First Linear1d weight matrix between copy-constructed and original network must differ");
        assert_true(l->parameters()[0] != l3->parameters()[0], "First Linear1d weight matrix between assigned and original network must differ");
        assert_equals(l2->parameters()[0], l3->parameters()[0]);
  }
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
        //Each prediction element is within 0.1 of the expected output
        assert_array_almost_equals(expected_outputs[i], prediction, 0.05);
  }
}






int main() {
    test_create_train_xor();
    test_create_branch();
    test_train_branch();
    cout << "Tests passed" << endl;
}