#include "Network.h"
#include <iostream>
#include <cassert>
#include <fstream>
#include <string>
#include <sstream>


Network::Network(const std::vector<unsigned>& topology)
{
	// number of layers
	unsigned numLayers = topology.size();
	
	for (unsigned layerNum = 0; layerNum < numLayers; layerNum++)
	{
		// add a new layer to the network
		mLayers.push_back(Layer());
		
		// if last layer, no output weights
		// otherwise it it the number of neurons in the next layer
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		// fill the layer with neurons, and add a bias neuron to the layer
		// (number of nerurons + 1(bias))
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++)
		{
			mLayers.back().push_back(Neuron(numOutputs, neuronNum));
			//std::cout << "Made a neuron!" << std::endl;
		}
	}
	
	// force the bias node's output value to 1.0
	mLayers.back().back().setOutputVal(1.0);
}

void Network::feedForward(const std::vector<double>& input)
{
	// check number of inputs
	assert(input.size() == mLayers[0].size() - 1);

	// assign input values to input neurons
	for (unsigned i = 0; i < input.size(); i++)
	{
		// input layer, ith neuron
		mLayers[0][i].setOutputVal(input[i]);
	}

	// forward propagate
	for (unsigned layer = 1; layer < mLayers.size(); layer++)
	{
		Layer& prevLayer = mLayers[layer - 1];
		
		for (unsigned neuron = 0; neuron < mLayers[layer].size() - 1; neuron++)
		{
			mLayers[layer][neuron].feedForward(prevLayer);
		}
	}
}

void Network::backPropagate(const std::vector<double>& target)
{
	// calculate overall net error (RMS of output neuron errors)
	Layer& outputLayer = mLayers.back();
	mError = 0.0;
	
	for (unsigned n = 0; n < outputLayer.size() - 1; n++)
	{
		double delta = target[n] - outputLayer[n].getOutputVal();
		mError += delta * delta;
	}

	mError /= outputLayer.size() - 1; // get average error squared
	mError = sqrt(mError); // RMS

	// implement a recent average measurement
	mRecentAverageError = (mRecentAverageError * mRecentAverageSmoothingFactor + mError) / (mRecentAverageSmoothingFactor + 1.0);

	// calc output layer gradients
	for (unsigned n = 0; n < outputLayer.size() - 1; n++)
	{
		outputLayer[n].calcOutputGradients(target[n]);
	}

	// calc hidden layer gradients
	for (unsigned layer = mLayers.size() - 2; layer > 0; layer--)
	{
		Layer& hiddenLayer = mLayers[layer];
		Layer& nextLayer = mLayers[layer + 1];
		
		for (unsigned n = 0; n < hiddenLayer.size(); n++)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}
	
	// for all layers from outputs to first hidden layer
	// update connection weights
	
	for (unsigned layer = mLayers.size() - 1; layer > 0; layer--)
	{
		Layer& currLayer = mLayers[layer];
		Layer& prevLayer = mLayers[layer - 1];

		for (unsigned n = 0; n < currLayer.size() - 1; n++)
		{
			currLayer[n].updateInputWeights(prevLayer);
		}
	}
}

void Network::getResults(std::vector<double>& result) const
{
	result.clear();

	for (unsigned n = 0; n < mLayers.back().size() - 1; n++)
	{
		result.push_back(mLayers.back()[n].getOutputVal());
	}
}

void Network::Draw(sf::RenderWindow& window)
{
	int x = 0;
	int y = 0;
	int xSpacing = SCREEN_WIDTH / mLayers.size() / 2;
	int ySpacing = radius * 2.5f;

	int maxDraw = 20;


	x = (window.getSize().x / 2) - ((mLayers.size() * xSpacing) / 2);
		// draw connections
	for (unsigned layer = 0; layer < mLayers.size() - 1; layer++)
	{
		
		int layerSize = mLayers[layer].size() < maxDraw ? mLayers[layer].size() : maxDraw;
		
		// center the layer on the screen y
		y = (window.getSize().y / 2) - ((layerSize * ySpacing) / 2);
		
		for (unsigned neuron = 0; neuron < layerSize - 1; neuron++)
		{

			int nextLayerSize = mLayers[layer + 1].size() < maxDraw ? mLayers[layer + 1].size() : maxDraw;

			if (layerSize == maxDraw)
			{
				if (neuron == layerSize - 2)
				{
					if (mLayers[layer].size() - layerSize  > 0)
					{
						layerSize++;
					}
				}
			}

			// connect to each neuron in the next layer
			for (unsigned nextNeuron = 0; nextNeuron < nextLayerSize - 1; nextNeuron++)
			{


				if (nextLayerSize == maxDraw)
				{
					if (nextNeuron == nextLayerSize - 2)
					{
						if (mLayers[layer].size() - nextLayerSize > 0)
						{
							nextLayerSize++;
						}
					}
				}

				
				sf::Vector2f start(x + radius, y + radius);
				// next layer top y
				float y2 = (window.getSize().y / 2) - ((nextLayerSize * ySpacing) / 2);
				sf::Vector2f end(x + (xSpacing)+radius, y2  + (nextNeuron * ySpacing) + radius);
				mLayers[layer][neuron].DrawConnections(window, start, end, nextNeuron);


				
			}

			y += ySpacing;
		}

		x += xSpacing;
		y = 0;
		
	}

	// center the layer on the screen x
	x = (window.getSize().x / 2) - ((mLayers.size() * xSpacing) / 2);
	
	for (unsigned layer = 0; layer < mLayers.size(); layer++)
	{
		int layerSize = mLayers[layer].size() < maxDraw ? mLayers[layer].size() : maxDraw;
		
		// center the layer on the screen y
		y = (window.getSize().y / 2) - ((layerSize * ySpacing) / 2);
		

		


		for (unsigned neuron = 0; neuron < layerSize - 1; neuron++)
		{

			sf::Vector2f position(x, y);
			
			mLayers[layer][neuron].Draw(window, position);

			y += ySpacing;

			position.y = y;

			if (layerSize == maxDraw)
			{
				if (neuron == layerSize - 2)
				{
					if (mLayers[layer].size() - layerSize > 0)
					{
						int num = mLayers[layer].size() - layerSize;
						std::string str = "+" + std::to_string(num);

						mLayers[layer][neuron].Draw(window, position, str);

						y += ySpacing;
						continue;
					}
				}
			}
		}
		
		x += xSpacing;
		y = 0;
	}






	
}

void Network::Serialize()
{
	std::ofstream file;
	file.open("network.txt");

	std::string layerDimensions = "Dimensions: ";
	
	for (unsigned layer = 0; layer < mLayers.size(); layer++)
	{
		layerDimensions += std::to_string(mLayers[layer].size()) + " ";
	}

	file << layerDimensions << std::endl;

	file << "weights: " << std::endl;

	for (unsigned layer = 0; layer < mLayers.size(); layer++)
	{
		for (unsigned neuron = 0; neuron < mLayers[layer].size(); neuron++)
		{
			mLayers[layer][neuron].Serialize(file);
		}
	}

	file.close();
}

Network* Network::Deserialize(std::string fileName)
{

	
	std::ifstream file;
	file.open(fileName);

	std::string line;
	std::getline(file, line);

	std::vector<std::string> dimensions;
	std::stringstream ss(line);
	std::string token;

	while (std::getline(ss, token, ' '))
	{
		dimensions.push_back(token);
	}

	std::vector<unsigned> layerDimensions;

	for (unsigned i = 1; i < dimensions.size(); i++)
	{
		layerDimensions.push_back(std::stoi(dimensions[i]));
	}

	std::vector<unsigned> topology;
		
	// create the network
	for (unsigned i = 0; i < layerDimensions.size(); i++)
	{
		topology.push_back(layerDimensions[i] - 1);
	}

	Network* newNetwork = new Network(topology);


	std::getline(file, line);
	
	for (unsigned layer = 0; layer < newNetwork->mLayers.size(); layer++)
	{
		for (unsigned neuron = 0; neuron < newNetwork->mLayers[layer].size(); neuron++)
		{
			newNetwork->mLayers[layer][neuron].Deserialize(file);
		}
	}

	file.close();

	return newNetwork;
}
