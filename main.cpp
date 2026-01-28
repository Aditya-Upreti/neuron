#include <iostream>
#include <cmath>
#include <vector>
#include <random>











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
        }else{
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
        matrix *t = new matrix(this->c,this->r,0);
            for (int i = 0; i < this->r; i++)
            {
                for (int j = 0; j < this->c; j++)
                {
                    t->setVal(j,i,this->getVal(i,j));
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

    void print(){
        
            for (int i = 0; i < this->r; i++)
            {
                for (int j = 0; j < this->c; j++)
                {
                    std::cout << this->getVal(i,j) << "\t\t";
                }
                std::cout<< "\n\n";
            }
    }
    int getRows(){
        return this->r;
    }
    int getCols(){
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
    void setVal(int i, double val){
        this->neurons.at(i)->setVal(val);
    }

    matrix *matrixifyVals(){
        matrix *m = new matrix(1, this->neurons.size(), false);
        for(int i=0;i<this->neurons.size();i++){
            m->setVal(0,i,this->neurons.at(i)->getVal());
        }
        return m;
    }
    matrix *matrixifyActivatedVals(){
        matrix *m = new matrix(1, this->neurons.size(), false);
        for(int i=0;i<this->neurons.size();i++){
            m->setVal(0,i,this->neurons.at(i)->getActivatedVal());
        }
        return m;
    }
    matrix *matrixifyDerivedVals(){
        matrix *m = new matrix(1, this->neurons.size(), false);
        for(int i=0;i<this->neurons.size();i++){
            m->setVal(0,i,this->neurons.at(i)->getDerivedVal());
        }
        return m;
    }

public:
    int size;
    std::vector<Neuron *> neurons;
};










matrix *multMatrix(matrix *a, matrix*b){
    if(a->getCols() != b->getRows()){
        std::cerr << "A Cols: " << a->getCols() << "!= B Rows: " << b->getRows() << std::endl;
        // assert(false);
    }

    matrix *c = new matrix(a->getRows(),b->getCols(), false);

    for(int i=0; i<a->getRows();i++){
        for(int j=0; j < b->getCols();j++){
            for(int k=0; k < b->getRows();k++){
                double p = a->getVal(i,k) * b->getVal(k,j);
                double newVal = c->getVal(i,j) + p;
                c->setVal(i,j,newVal);
            }
        }
    }

    return c;
}


std::vector<double> matrixToVector(matrix *a){
    std::vector<double> result;
    for(int i = 0; i<a->getRows();i++){
         for(int j = 0; j<a->getRows();j++){
            result.push_back(a->getVal(i,j));
         }
    }
    return result;
}













class neuralNetwork{
    public:
        neuralNetwork(std::vector<int> topology){
            this->topology = topology;
            this->topoSize = topology.size();

            for(int i=0;i<this->topoSize; i++){
                layer *l = new layer(topology.at(i));
                this->layers.push_back(l);
            }
            for(int i=0;i<this->topoSize-1; i++){
                matrix *m = new matrix(topology.at(i),topology.at(i+1),true);
                this->weightmatrices.push_back(m);
                // std::cout<<"yes\n";

            }
        }

        matrix *getNeuronMatrix(int i){
            return this->layers.at(i)->matrixifyVals();
        }
        matrix *getAditivatedNeuronMatrix(int i){
            return this->layers.at(i)->matrixifyActivatedVals();
        }
        matrix *getDerivedNeuronMatrix(int i){
            return this->layers.at(i)->matrixifyDerivedVals();
        }


        void setCurrentInput(std::vector<double> input){
            this->input = input;

            for(int i =0;i<input.size();i++){
                this->layers.at(0)->setVal(i,input.at(i));
            }
        }
        void print(){

            for(int i=0;i<this->layers.size();i++){
                std::cout << "LAYER: " << i << std::endl;
                if(i==0){
                     matrix *m = this->layers.at(i)->matrixifyVals();
                     m->print();
                }else{
                     matrix *m = this->layers.at(i)->matrixifyActivatedVals();
                     m->print();

                }
                std::cout<< "============================================================\n";
                if(i<this->layers.size()-2){
                    std::cout<< "Weight Matrix: " << i <<"\n";
                    this->getWeightMatrix(i)->print();
                }
                std::cout<< "============================================================\n";
            }
        }
        matrix *getWeightMatrix(int i){return this->weightmatrices.at(i);}

        void feedForward(){

            for(int i=0;i<this->layers.size()-1;i++){
                matrix *a = this->getNeuronMatrix(i);
                if(i != 0){
                    a=this->getAditivatedNeuronMatrix(i);
                }
                matrix *b = this->getWeightMatrix(i);
                matrix *c = multMatrix(a,b);
                std::vector<double> vals;
                for(int j = 0; j<c->getCols(); j++){
                    vals.push_back(c->getVal(0,j));
                    this->setNeuronValue(i+1,j, c->getVal(0,j));
                }
            }
        }
        void setNeuronValue(int indexlayer, int indexNeuron, double val){this ->layers.at(indexlayer)->setVal(indexNeuron, val);}
    private:
        int topoSize;
        std::vector<int> topology;
        std::vector<matrix *> weightmatrices;
        std::vector<layer *> layers;
        std::vector<double> input;
        
};


























int main(int argc, char **argv)
{

    // Neuron *n = new Neuron(0.9);
    // matrix *m = new matrix(3,2,true);
    // m->print();
    // std::cout << "\n----------------------------------\n\n";
    // m->transpose()->print();

    std::vector<double> input = {1,0,1};
    std::vector<int> topology = {3,2,1};
    neuralNetwork *nn = new neuralNetwork(topology);
    nn->setCurrentInput(input);
    nn->feedForward();
    nn->print();


    return 0;
}