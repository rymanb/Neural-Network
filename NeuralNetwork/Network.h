#pragma once

#include <SFML/Graphics.hpp>
#include "Definitions.h"
#include "Neuron.h"

typedef std::vector<Neuron> Layer;

class Network
{
public:
	Network(const std::vector<unsigned>& topology);
	void feedForward(const std::vector<double>& input);
	void backPropagate(const std::vector<double>& target);
	void getResults(std::vector<double>& result) const;
	double getRecentAverageError() const { return mRecentAverageError; }

	void Draw(sf::RenderWindow& window);

	void Serialize();
	static Network* Deserialize(std::string fileName);

	
	
private:
	std::vector<Layer> mLayers;
	double mError;
	double mRecentAverageError;
	double mRecentAverageSmoothingFactor;

};

