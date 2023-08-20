#include "Neuron.h"
#include <assert.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>

double Neuron::eta = 0.15; // [0.0..1.0] overall net training rate
double Neuron::alpha = 0.5; // [0.0..n] multiplier of last weight change (momentum)

Neuron::Neuron(unsigned numOutputs, unsigned index) : mOutputVal(0), mNeuronIndex(index)
{
	for (unsigned c = 0; c < numOutputs; c++)
	{
		mOutputWeights.push_back(Connection());
		mOutputWeights.back().weight = randomWeight();
	}
}

void Neuron::feedForward(const Layer& prevLayer)
{
	double sum = 0.0;

	// sum the previous layer's outputs (which are our inputs)
	// include the bias node from the previous layer
	for (unsigned n = 0; n < prevLayer.size(); n++)
	{
		double output = prevLayer[n].getOutputVal();

		sum += output * prevLayer[n].mOutputWeights[mNeuronIndex].weight;
	}

	sum /= prevLayer.size();


	mOutputVal = Neuron::activationFunction(sum);

}

void Neuron::calcOutputGradients(double targetVal)
{
	// difference between target and actual output
	double delta = targetVal - mOutputVal;
	// derivative of activation function
	mGradient = delta * Neuron::activationFunctionDerivative(mOutputVal);
}

void Neuron::calcHiddenGradients(const Layer& nextLayer)
{
	double dow = sumDOW(nextLayer);
	mGradient = dow * Neuron::activationFunctionDerivative(mOutputVal);
}

void Neuron::updateInputWeights(Layer& prevLayer)
{
	// the weights to be updated are in the Connection container
	// in the neurons in the preceding layer

	for (unsigned n = 0; n < prevLayer.size(); n++)
	{
		Neuron& neuron = prevLayer[n];
		double oldDeltaWeight = neuron.mOutputWeights[mNeuronIndex].deltaWeight;
		
		double newDeltaWeight =
			// individual input, magnified by the gradient and train rate
			eta
			* neuron.getOutputVal()
			* mGradient
			// also add momentum = a fraction of the previous delta weight
			+ alpha
			* oldDeltaWeight;
		
		neuron.mOutputWeights[mNeuronIndex].deltaWeight = newDeltaWeight;
		neuron.mOutputWeights[mNeuronIndex].weight += newDeltaWeight;
   }
}

void Neuron::Draw(sf::RenderWindow& window, sf::Vector2f position, std::string str)
{
	sf::Color color = sf::Color(255 * mOutputVal, 255 * mOutputVal, 255 * mOutputVal);
	
	// invers color of circle
	sf::Color textColor = sf::Color(255 - color.r, 255 - color.g, 255 - color.b);



	sf::CircleShape circle(radius);
	circle.setFillColor(color);


	// outline thickness
	circle.setOutlineThickness(5);
	// outline color is grey 
	sf::Color grey(100, 100, 100);
	circle.setOutlineColor(grey);
	
	
	circle.setPosition(position);
	window.draw(circle);

	sf::Text text;
	// use arial font
	sf::Font font;
	font.loadFromFile("arial.ttf");
	text.setFont(font);

	if (str == "")
	{

		// round the output value to 2 decimal places
		std::stringstream ss;
		ss << std::fixed << std::setprecision(2) << mOutputVal;
		text.setString(ss.str());
	}
	else
	{
		text.setString(str);
	}
	text.setCharacterSize(radius / 1.5);
	text.setFillColor(textColor);
	
	// center the text
	sf::FloatRect textRect = text.getLocalBounds();
	text.setOrigin(textRect.left + textRect.width / 2.0f,
		textRect.top + textRect.height / 2.0f);
	text.setPosition(position.x + radius, position.y + radius);
	
	
	window.draw(text);
	
	
}

void Neuron::DrawConnections(sf::RenderWindow& window, sf::Vector2f start, sf::Vector2f end, int weightInd)
{
	sf::Vertex line[] =
	{
		sf::Vertex(start),
		sf::Vertex(end)
	};

	float weight = mOutputWeights[weightInd].weight;

	// line color is based on the weight, red is positive, blue is negative
	sf::Color color = sf::Color(255 * weight, 0, 255 * (1 - weight));
	line[0].color = color;
	line[1].color = color;
	
	

	window.draw(line, 2, sf::Lines);
}

void Neuron::Serialize(std::ofstream& file)
{
	for (unsigned i = 0; i < mOutputWeights.size(); i++)
	{
		std::string str = std::to_string(mOutputWeights[i].weight);
		file << str << std::endl;

		
	}
}

void Neuron::Deserialize(std::ifstream& file)
{
	for (unsigned i = 0; i < mOutputWeights.size(); i++)
	{
		std::string str;
		file >> str;
		mOutputWeights[i].weight = std::stod(str);
	}
}


double Neuron::activationFunction(double x)
{
	//return tanh(x);

	// sigmoid function
	return 1 / (1 + exp(-x));
	
}

double Neuron::activationFunctionDerivative(double x)
{
	// tanh derivative
	//return 1.0 - x * x;

	// sigmoid derivative
	return x * (1 - x);
}

double Neuron::sumDOW(const Layer& nextLayer) const
{
	double sum = 0.0;

	// sum our contributions of the errors at the nodes we feed

	for (unsigned n = 0; n < nextLayer.size() - 1; n++)
	{
		sum += mOutputWeights[n].weight * nextLayer[n].mGradient;
	}

	return sum;
}
