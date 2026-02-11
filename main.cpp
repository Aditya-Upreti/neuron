#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <cassert>
#include <algorithm>

class Neuron
{
public:
    // fast digmoid function
    //  f = x/(1+|x|)
    void activate()
    {
        this->activatedVal = this->val / (1 + abs(this->val));
    }

    // derivative for fast sigmoid function
    //  f' = f + (1-f)
    void derive()
    {

        this->derivedVal = this->activatedVal * (1 - this->activatedVal);
    }
    // getter
    double getVal() { return this->val; }
    void setVal(double val)
    {
        this->val = val;
        activate();
        derive();
    }
    double getActivatedVal() { return this->activatedVal; }
    double getDerivedVal() { return this->derivedVal; }

    Neuron(double val)
    {
        this->val = val;
        activate();
        derive();
    }

private:
    double val;
    double activatedVal;
    double derivedVal;
};

class matrix
{
public:
    matrix(int r, int c, bool isRandom)
    {
        this->r = r;
        this->c = c;

        if (isRandom)
        {
            for (int i = 0; i < r; i++)
            {
                std::vector<double> cols;
                for (int j = 0; j < c; j++)
                {
                    cols.push_back(this->generatRandomNo());
                }
                this->values.push_back(cols);
            }
        }
        else
        {
            for (int i = 0; i < r; i++)
            {
                std::vector<double> cols(c);
                for (int j = 0; j < c; j++)
                {
                    cols.push_back(0);
                }
                this->values.push_back(cols);
            }
        }
    }
    matrix *transpose()
    {
        matrix *t = new matrix(this->c, this->r, 0);
        for (int i = 0; i < this->r; i++)
        {
            for (int j = 0; j < this->c; j++)
            {
                t->setVal(j, i, this->getVal(i, j));
            }
        }
        return t;
    }
    void setVal(int r, int c, double v)
    {
        this->values.at(r).at(c) = v;
    }
    double getVal(int r, int c)
    {
        return this->values.at(r).at(c);
    }
    double generatRandomNo()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);
        return dis(gen);
    }

    void print()
    {

        for (int i = 0; i < this->r; i++)
        {
            for (int j = 0; j < this->c; j++)
            {
                std::cout << this->getVal(i, j) << "\t\t";
            }
            std::cout << "\n\n";
        }
    }
    int getRows()
    {
        return this->r;
    }
    int getCols()
    {
        return this->c;
    }

private:
    int r;
    int c;
    std::vector<std::vector<double>> values;
};

class layer
{
public:
    layer(int size)
    {
        this->size = size;
        for (int i = 0; i < size; i++)
        {
            Neuron *n = new Neuron(0.00);
            this->neurons.push_back(n);
        }
    }
    void setVal(int i, double val)
    {
        this->neurons.at(i)->setVal(val);
    }

    matrix *matrixifyVals()
    {
        matrix *m = new matrix(1, this->neurons.size(), false);
        for (int i = 0; i < this->neurons.size(); i++)
        {
            m->setVal(0, i, this->neurons.at(i)->getVal());
        }
        return m;
    }
    matrix *matrixifyActivatedVals()
    {
        matrix *m = new matrix(1, this->neurons.size(), false);
        for (int i = 0; i < this->neurons.size(); i++)
        {
            m->setVal(0, i, this->neurons.at(i)->getActivatedVal());
        }
        return m;
    }
    matrix *matrixifyDerivedVals()
    {
        matrix *m = new matrix(1, this->neurons.size(), false);
        for (int i = 0; i < this->neurons.size(); i++)
        {
            m->setVal(0, i, this->neurons.at(i)->getDerivedVal());
        }
        return m;
    }
    std::vector<Neuron *> getNeurons() { return this->neurons; };

public:
    int size;
    std::vector<Neuron *> neurons;
};

matrix *multMatrix(matrix *a, matrix *b)
{
    if (a->getCols() != b->getRows())
    {
        std::cerr << "A Cols: " << a->getCols() << "!= B Rows: " << b->getRows() << std::endl;
        // assert(false);
    }

    matrix *c = new matrix(a->getRows(), b->getCols(), false);

    for (int i = 0; i < a->getRows(); i++)
    {
        for (int j = 0; j < b->getCols(); j++)
        {
            for (int k = 0; k < b->getRows(); k++)
            {
                double p = a->getVal(i, k) * b->getVal(k, j);
                double newVal = c->getVal(i, j) + p;
                c->setVal(i, j, newVal);
            }
        }
    }

    return c;
}

std::vector<double> matrixToVector(matrix *a)
{
    std::vector<double> result;
    for (int i = 0; i < a->getRows(); i++)
    {
        for (int j = 0; j < a->getRows(); j++)
        {
            result.push_back(a->getVal(i, j));
        }
    }
    return result;
}

class neuralNetwork
{
public:
    neuralNetwork(std::vector<int> topology)
    {
        this->topology = topology;
        this->topoSize = topology.size();

        for (int i = 0; i < this->topoSize; i++)
        {
            layer *l = new layer(topology.at(i));
            this->layers.push_back(l);
        }
        for (int i = 0; i < this->topoSize - 1; i++)
        {
            matrix *m = new matrix(topology.at(i), topology.at(i + 1), true);
            this->weightmatrices.push_back(m);
            // std::cout<<"yes\n";
        }
    }

    matrix *getNeuronMatrix(int i)
    {
        return this->layers.at(i)->matrixifyVals();
    }
    matrix *getAditivatedNeuronMatrix(int i)
    {
        return this->layers.at(i)->matrixifyActivatedVals();
    }
    matrix *getDerivedNeuronMatrix(int i)
    {
        return this->layers.at(i)->matrixifyDerivedVals();
    }

    void setCurrentInput(std::vector<double> input)
    {
        this->input = input;

        for (int i = 0; i < input.size(); i++)
        {
            this->layers.at(0)->setVal(i, input.at(i));
        }
    }
    void setCurrentTarget(std::vector<double> target)
    {
        this->target = target;
    }
    void print()
    {

        for (int i = 0; i < this->layers.size(); i++)
        {
            std::cout << "LAYER: " << i << std::endl;
            if (i == 0)
            {
                matrix *m = this->layers.at(i)->matrixifyVals();
                m->print();
            }
            else
            {
                matrix *m = this->layers.at(i)->matrixifyActivatedVals();
                m->print();
            }
            std::cout << "============================================================\n";
            if (i < this->layers.size() - 2)
            {
                std::cout << "Weight Matrix: " << i << "\n";
                this->getWeightMatrix(i)->print();
            }
            std::cout << "============================================================\n";
        }
    }
    matrix *getWeightMatrix(int i) { return this->weightmatrices.at(i); }

    void feedForward()
    {

        for (int i = 0; i < this->layers.size() - 1; i++)
        {
            matrix *a = this->getNeuronMatrix(i);
            if (i != 0)
            {
                a = this->getAditivatedNeuronMatrix(i);
            }
            matrix *b = this->getWeightMatrix(i);
            matrix *c = multMatrix(a, b);
            std::vector<double> vals;
            for (int j = 0; j < c->getCols(); j++)
            {
                vals.push_back(c->getVal(0, j));
                this->setNeuronValue(i + 1, j, c->getVal(0, j));
            }
        }
    }
    void setNeuronValue(
        int indexlayer, int indexNeuron, double val)
    {
        this->layers.at(indexlayer)->setVal(indexNeuron, val);
    }
    double getTotalError() { return this->error; }
    std::vector<double> getErrors() { return this->errors; }

    void setErrors()
    {
        if (this->target.size() == 0)
        {
            std::cerr << "no target for this neural network " << std::endl;
            assert(false);
        }
        if (this->target.size() != this->layers.at(this->layers.size() - 1)->getNeurons().size())
        {
            std::cerr << "Target size is not same as output layer size: " << this->layers.at(this->layers.size() - 1)->getNeurons().size() << std::endl;
            assert(false);
        }
        this->error = 0.00;
        int outputLayerIndex = this->layers.size() - 1;
        std::vector<Neuron *> outputNeurons = this->layers.at(outputLayerIndex)->getNeurons();

        for (int i = 0; i < this->target.size(); i++)
        {

            double tempErr = (outputNeurons.at(i)->getActivatedVal() - this->target.at(i));
            this->errors.push_back(tempErr);
            this->error += tempErr;
        }
        historicalErrors.push_back(this->error);
    }

    void backProgation()
    {
        
        matrix *gradient;
        std::vector<matrix *> newWeights;
        int outputLayerIndex = this->layers.size() - 1;
        matrix *derivedValuesYtoZ = this->layers.at(outputLayerIndex)->matrixifyDerivedVals();
        matrix *gradientsYtoZ = new matrix(1, this->layers.at(outputLayerIndex)->getNeurons().size(),false);
        
        for (int i = 0; i < this->errors.size(); i++)
        {
            
            double d = derivedValuesYtoZ->getVal(0, i);
            std::cout<<"00000\n";
            double e = this->errors.at(i);
            double g = d * e;
            gradientsYtoZ->setVal(0, i, g);
        }
        int lastHiddenLayerIndex = outputLayerIndex - 1;
        layer *lastHiddenLayer = this->layers.at(lastHiddenLayerIndex);
        matrix *weightsOutputToHidden = this->weightmatrices.at(outputLayerIndex - 1);
        matrix *deltaOutputToHidden = multMatrix(gradientsYtoZ->transpose(), lastHiddenLayer->matrixifyActivatedVals())->transpose();

        matrix *newWeightOutputToHidden = new matrix(deltaOutputToHidden->getRows(), deltaOutputToHidden->getCols(), false);

        for (int r = 0; r < deltaOutputToHidden->getRows(); r++)
        {
            for (int c = 0; c < deltaOutputToHidden->getCols(); c++)
            {
                double originalWeight = weightsOutputToHidden->getVal(r, c);
                double deltaWeight = deltaOutputToHidden->getVal(r, c);
                newWeightOutputToHidden->setVal(r, c, (originalWeight - deltaWeight));
            }
        }
        newWeights.push_back(newWeightOutputToHidden);

        gradient = new matrix(gradientsYtoZ->getRows(), gradientsYtoZ->getCols(), false);
        for (int r = 0; r < gradientsYtoZ->getRows(); r++)
        {
            for (int c = 0; c < gradientsYtoZ->getCols(); c++)
            {
                gradient->setVal(r, c, gradientsYtoZ->getVal(r,c));
            }
        }





        for (int i = (outputLayerIndex - 1); i > 0; i--)
        {
            layer *l = this->layers.at(i);
            matrix *derivedHidden = l->matrixifyDerivedVals();
            matrix *activatedHidden = l->matrixifyActivatedVals();
            matrix *derivedGradients = new matrix(1, l->getNeurons().size(), false);
            matrix *weightMatrix = this->weightmatrices.at(i);
            matrix *OriginalweightMatrix = this->weightmatrices.at(i - 1);
            for (int r = 0; r < derivedGradients->getRows(); r++)
            {
                double sum = 0;
                for (int c = 0; c < derivedGradients->getCols(); c++)
                {
                    double p = gradient->getVal(0, c) * weightMatrix->getVal(r, c);
                    sum += p;
                }
                double g = sum * activatedHidden->getVal(0, r);
                derivedGradients->setVal(0, r, g);
            }
            
            matrix *leftNeurons = (i - 1) == 0 ? this->layers.at(i - 1)->matrixifyVals() : this->layers.at(i - 1)->matrixifyActivatedVals();
            matrix *deltaWeights = multMatrix(derivedGradients->transpose(), leftNeurons)->transpose();
            matrix *newWeightsHidden = new matrix(deltaWeights->getRows(), deltaWeights->getCols(), false);

            for (int r = 0; r < newWeightsHidden->getRows(); r++)
            {
                for (int c = 0; c < newWeightsHidden->getCols(); c++)
                {
                    double w = OriginalweightMatrix->getVal(r, c);

                    double d = deltaWeights->getVal(r, c);
                    
                    double n = w - d;
                    newWeightsHidden->setVal(r, c, n);
                }
            }
            
            gradient = new matrix(derivedGradients->getRows(), derivedGradients->getCols(), false);
            for (int r = 0; r < derivedGradients->getRows(); r++)
            {
                for (int c = 0; c < derivedGradients->getCols(); c++)
                {
                    gradient->setVal(r, c, derivedGradients->getVal(r, c));
                }
            }
            
        // std::cout << "hello\n";
            newWeights.push_back(newWeightsHidden);
        }
        // std::cout << "New weight size : " << newWeights.size() << std::endl;
        // std::cout << "old weight size : " << this->weightmatrices.size() << std::endl;
        // for(int i=0;i<newWeights.size();i++){
        //     cout << i << ":\n";
        //     newWeights.at(i)->print();
        // }
        std::reverse(newWeights.begin(), newWeights.end());
        this->weightmatrices = newWeights;
    }

private:
    int topoSize;
    std::vector<int> topology;
    std::vector<matrix *> weightmatrices;
    std::vector<layer *> layers;
    std::vector<double> input;
    std::vector<double> target;
    double error;
    std::vector<double> errors;
    std::vector<double> historicalErrors;
};

int main(int argc, char **argv)
{

    // Neuron *n = new Neuron(0.9);
    // matrix *m = new matrix(3,2,true);
    // m->print();
    // std::cout << "\n----------------------------------\n\n";
    // m->transpose()->print();

    std::vector<double> input = {1, 0, 1};
    std::vector<int> topology = {3, 2, 3};
    neuralNetwork *nn = new neuralNetwork(topology);
    nn->setCurrentInput(input);
    nn->setCurrentTarget(input);

    // training

    for (int i = 0; i < 5; i++)
    {
        std::cout << "Epoc" << i << "\n";
        nn->feedForward();
        nn->setErrors();

        std::cout << "Total Error: " << nn->getTotalError() << std::endl;

        nn->backProgation();
    }

    return 0;
}