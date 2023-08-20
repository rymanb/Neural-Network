#pragma once
#include <SFML/Graphics.hpp>
#include "Definitions.h"
#include <fstream>

class Neuron;
typedef std::vector<Neuron> Layer;

struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned index);
	void feedForward(const Layer& prevLayer);
	void setOutputVal(double val) { mOutputVal = val; }
	double getOutputVal() const { return mOutputVal; }
	std::vector<Connection> GetWeights() { return mOutputWeights; }
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer& nextLayer);
	void updateInputWeights(Layer& prevLayer);

	void Draw(sf::RenderWindow& window, sf::Vector2f position, std::string text = "");
	void DrawConnections(sf::RenderWindow& window, sf::Vector2f position1, sf::Vector2f position2, int weightInd);

	void Serialize(std::ofstream& file);
	void Deserialize(std::ifstream& file);
	
private:

	static double eta; // [0.0..1.0] overall net training rate
	static double alpha; // [0.0..n] multiplier of last weight change (momentum)
	
	static double randomWeight() { return rand() / double(RAND_MAX); }
	static double activationFunction(double x);
	static double activationFunctionDerivative(double x);
	double sumDOW(const Layer& nextLayer) const;

	unsigned mNeuronIndex;
	double mOutputVal;
	double mGradient;
	
	std::vector<Connection> mOutputWeights;
	
};

