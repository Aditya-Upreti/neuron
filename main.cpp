#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

// --- 1. Helper Functions (The Math) ---

// Sigmoid Function: Squishes numbers to be between 0 and 1
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of Sigmoid: Used for Backpropagation (The "Sensitivity")
// If y = sigmoid(x), then derivative = y * (1 - y)
double sigmoidDerivative(double y) {
    return y * (1.0 - y);
}

// --- 2. The Neuron and Connection Structs ---

struct Connection {
    double weight;
    double deltaWeight;
};

class Neuron; // Forward declaration

typedef vector<Neuron> Layer;

// --- 3. The Neuron Class ---
class Neuron {
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal() const { return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    static double randomWeight() { return rand() / double(RAND_MAX); }
    double m_outputVal;
    vector<Connection> m_outputWeights; // Weights to the NEXT layer
    unsigned m_myIndex;
    double m_gradient;
    double eta = 0.15; // Learning Rate (How fast we learn)
    double alpha = 0.5; // Momentum (Keeps us moving in the right direction)
};

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
}

void Neuron::feedForward(const Layer &prevLayer) {
    double sum = 0.0;
    // Loop through the previous layer's outputs and multiply by weights
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    m_outputVal = sigmoid(sum);
}

void Neuron::calcOutputGradients(double targetVal) {
    double delta = targetVal - m_outputVal; // Error: Target - Output
    m_gradient = delta * sigmoidDerivative(m_outputVal);
}
void Neuron::calcHiddenGradients(const Layer &nextLayer) {
    double dow = 0.0; // Sum of derivatives of weights

    // Sum up the contributions of the errors sent to the next layer
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        // CORRECTED LINE:
        // We use OUR weights (m_outputWeights) that connect to the next layer
        // Multiplied by the Next Layer's gradients.
        dow += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    m_gradient = dow * sigmoidDerivative(m_outputVal);
}

void Neuron::updateInputWeights(Layer &prevLayer) {
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        // The Magic Formula: NewWeight = OldWeight + (LearningRate * Gradient * Input)
        double newDeltaWeight =
                // Individual input, magnified by the gradient and train rate:
                eta * neuron.getOutputVal() * m_gradient
                // Also add momentum = a fraction of the previous delta weight
                + alpha * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

// --- 4. The Network Class (The Manager) ---
class Net {
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }

private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    double m_smoothingFactor = 100.0;
};

Net::Net(const vector<unsigned> &topology) {
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        // For each layer, create neurons + 1 bias neuron
        unsigned numOutputs = (layerNum == numLayers - 1) ? 0 : topology[layerNum + 1];
        
        // Add neurons to the layer
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
        }
        
        // Force the bias node's output to 1.0 (It's the constant)
        m_layers.back().back().setOutputVal(1.0);
    }
}

void Net::feedForward(const vector<double> &inputVals) {
    // Assign (latch) the input values into the input neurons
    for (unsigned i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // Forward propagation
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

void Net::backProp(const vector<double> &targetVals) {
    // Calculate overall net error (RMS of output neuron errors)
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= (outputLayer.size() - 1); // get average error squared
    m_error = sqrt(m_error); // RMS

    // Implement a recent average measurement:
    m_recentAverageError = (m_recentAverageError * m_smoothingFactor + m_error) / (m_smoothingFactor + 1.0);

    // Calculate Output Layer Gradients
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Calculate Hidden Layer Gradients
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // Update Connection Weights
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::getResults(vector<double> &resultVals) const {
    resultVals.clear();
    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

// --- 5. Main Function (The Training Loop) ---
int main() {
    srand(time(NULL));

    // Topology: 2 Inputs, 2 Hidden Neurons, 1 Output
    vector<unsigned> topology;
    topology.push_back(2);
    topology.push_back(2);
    topology.push_back(1);
    
    Net myNet(topology);

    // The XOR Data Table
    // { Input A, Input B, Target Output }
    vector<vector<double>> inputs = { {0,0}, {0,1}, {1,0}, {1,1} };
    vector<vector<double>> targets = { {0},   {1},   {1},   {0}   };

    cout << "Training..." << endl;

    // Train for 5000 Epochs
    for (int i = 0; i < 5000; ++i) {
        // Pick a random index (0 to 3)
        int idx = rand() % 4;
        
        myNet.feedForward(inputs[idx]);
        myNet.backProp(targets[idx]);
        
        if (i % 1000 == 0) {
            cout << "Epoch: " << i << " | Error: " << myNet.getRecentAverageError() << endl;
        }
    }

    cout << "\n--- Final Results ---" << endl;
    for (int i = 0; i < 4; ++i) {
        myNet.feedForward(inputs[i]);
        vector<double> results;
        myNet.getResults(results);
        
        cout << "Input: " << inputs[i][0] << ", " << inputs[i][1];
        cout << " | Output: " << results[0];
        cout << " | Target: " << targets[i][0] << endl;
    }

    return 0;
}