

#include <iostream>
#include <SFML/Graphics.hpp>
#include <chrono>
#include "Network.h"
#include "Definitions.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <string>
#include <ostream>
#include <sstream>

#include <windows.h>


// fopen
#include <stdio.h>

// random number generator
#include <random>

#pragma warning(disable:4996)


int main()
{
	bool training = false;
	
	std::vector<std::pair<std::vector<double>,std::vector<double>>> trainingData;
	std::vector<std::vector<double>> testData;

	const int dimensions = 28;

	// training data
	std::fstream file("mnist_test.csv");

	int i = 0;

	std::string line;
	while (std::getline(file, line))
	{
		std::vector<double> data;
		float label = 0;
		std::stringstream ss(line);
		std::string value;
		
		bool first = true;
		while (std::getline(ss, value, ','))
		{
			if (first)
			{
				if (value != "label")
				{
					label = std::stof(value);
				}
				first = false;
			}
			else
			{
				double val = std::stof(value);
				if (val > 0)
				{
					val = val / 255;
				}
				data.push_back(val);
			}

		}

		// convert label to one-hot vector
		std::vector<double> labelVector(10, 0);
		labelVector[label] = 1;
		
		trainingData.push_back(std::make_pair(labelVector, data));

		i++;
		//if (i > 100)
		//	break;
	}

	// remove first element
	trainingData.erase(trainingData.begin());


	


	std::vector<unsigned> topology;
	// input layer
	topology.push_back(dimensions * dimensions);

	// hidden layer
	topology.push_back(28);
	topology.push_back(10);
	
	
	
	// output layer (10 possible digits)
	topology.push_back(10);

	//topology.push_back(2);
	//topology.push_back(3);
	//topology.push_back(3);
	//topology.push_back(2);
	//topology.push_back(1);
	
	//Network myNet(topology);


	Network myNet = *Network::Deserialize("network.txt");

	// input values for XOR
	std::vector<std::vector<double>> inputPool = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
	std::vector<std::vector<double>> targetPool = { { 0 }, { 1 }, { 1 }, { 0 } };

	std::vector<double> input;
	std::vector<double> target;
	std::vector<double> results;

	// training per loop
	unsigned trainingPerLoop = 5000;
	

	std::chrono::steady_clock::time_point previous_time;


	sf::RenderWindow window(sf::VideoMode(SCREEN_WIDTH,  SCREEN_HEIGHT), "Neural networks", sf::Style::Close);
	window.setView(sf::View(sf::FloatRect(0, 0,SCREEN_WIDTH, SCREEN_HEIGHT)));

	sf::Event event;

	// image for displaying the number
	sf::Image image;
	image.create(dimensions, dimensions, sf::Color::White);
	
	sf::Text text;
	sf::Font font;
	font.loadFromFile("arial.ttf");
	text.setFont(font);
	text.setCharacterSize(24);
	text.setFillColor(sf::Color::White);
	text.setPosition(SCREEN_WIDTH / 2, 0);


	//Oh yeah, we're also gonna use rand().
	srand(std::chrono::system_clock::now().time_since_epoch().count());

	previous_time = std::chrono::steady_clock::now();

	double userInput[dimensions][dimensions] = { 0 };

	while (1 == window.isOpen())
	{
		std::chrono::microseconds delta_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - previous_time);


		while (1 == window.pollEvent(event))
		{
			switch (event.type)
			{
			case sf::Event::Closed:
			{
				window.close();

				break;
			}


			}


		}
		
		float error = 0.0f;

		unsigned index = 0;

		
		
		if (training)
		{
			// train the network
			for (unsigned i = 0; i < trainingPerLoop; i++)
			{
				// get random input and target
				index = rand() % trainingData.size();
				input = trainingData[index].second;
				target = { trainingData[index].first };

				// get random input and target
				//unsigned index = rand() % inputPool.size();
				//input = inputPool[index];
				//target = targetPool[index];

				// feed forward
				myNet.feedForward(input);

				// get results
				myNet.getResults(results);

				// back propagate
				//myNet.backPropagate(target);



				error += myNet.getRecentAverageError();
			}

			error /= trainingPerLoop;

			if (error < 0.002f)
			{
				std::cout << "Error: " << error << std::endl;
				std::cout << "Training complete!" << std::endl;
				//training = false;
			}
			

			
			std::string str = "Error: " + std::to_string(error);
			text.setString(str);

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
			{
				myNet.Serialize();
			}



		}
		else
		{

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::C))
			{
				for (unsigned i = 0; i < dimensions; i++)
				{
					for (unsigned j = 0; j < dimensions; j++)
					{
						userInput[i][j] = 0;
					}
				}
			}
			// if mouse is pressed over pixel, set pixel to black
			if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
			{
				const int brushSize = 2;
				
				sf::Vector2i mousePos = sf::Mouse::getPosition(window);
				int x = mousePos.x / 10;
				int y = mousePos.y / 10;

				if (x >= 0 && x < dimensions && y >= 0 && y < dimensions)
				{
					for (int i = -brushSize; i < brushSize; i++)
					{
						for (int j = -brushSize; j < brushSize; j++)
						{
							if (x + i >= 0 && x + i < dimensions && y + j >= 0 && y + j < dimensions)
							{
								// feather out the brush
								float dist = sqrt(i * i + j * j);
								float alpha = 1.0f - dist / brushSize;
								if (alpha < 0.0f)
									alpha = 0.0f;
								
								if (userInput[y + i][x + j] + abs(alpha) < 1)
								{
									userInput[y + i][x + j] += abs(alpha);
								}
								else
								{
									userInput[y + i][x + j] = 1;
								}

								
							}
						}
					}
					
				}
			}
			else if (sf::Mouse::isButtonPressed(sf::Mouse::Right))
			{
				sf::Vector2i mousePos = sf::Mouse::getPosition(window);
				int x = mousePos.x / 10;
				int y = mousePos.y / 10;

				if (x >= 0 && x < dimensions && y >= 0 && y < dimensions)
				{
					userInput[x][y] = 0.0f;

				}
			}



			std::vector<double> input;
			for (int x = 0; x < dimensions; x++)
			{
				for (int y = 0; y < dimensions; y++)
				{
					input.push_back(userInput[x][y]);
				}
			}

			// feed forward
			if (input.size())
				myNet.feedForward(input);

			static int acctual = 0;
			int predicted = 0;
			
			// get results
			myNet.getResults(results);
			
			// get the highest value
			double highest = 0.0f;
			for (unsigned i = 0; i < results.size(); i++)
			{
				if (results[i] > highest)
				{
					highest = results[i];
					predicted = i;
				}
			}
			
			// if user presses any number key, set the target to that number
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num0))
			{
				target = { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
				acctual = 0;
			}
			else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num1))
			{
				target = { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
				acctual = 1;
			}
			else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num2))
			{
				target = { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
				acctual = 2;
			}
			else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num3))
			{
				target = { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 };
				acctual = 3;
			}
			else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num4))
			{
				target = { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };
				acctual = 4;
			}
			else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num5))
			{
				target = { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
				acctual = 5;
			}
			else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num6))
			{
				target = { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 };
				acctual = 6;
			}
			else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num7))
			{
				target = { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 };
				acctual = 7;
			}
			else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num8))
			{
				target = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 };
				acctual = 8;
			}
			else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num9))
			{
				target = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
				acctual = 9;
			}
			
			// if the user presses space, train the network
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
			{
				myNet.backPropagate(target);
				std::cout << "acctual: " << acctual << " predicted: " << predicted << std::endl;
			}

			trainingData[0].second = input;
			index = 0;
			

		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
		{
			myNet.Serialize();
		}
		// print first element

		for (int x = 0; x < dimensions; x++)
		{
			for (int y = 0; y < dimensions; y++)
			{
				// array index = x + y * width
				int index2 = x + y * dimensions;
				if (trainingData[index].second[index2] > 0)
					image.setPixel(x, y, sf::Color(255 * trainingData[index].second[index2], 255 * trainingData[index].second[index2], 255 * trainingData[index].second[index2], 255));
				else
					image.setPixel(x, y, sf::Color(10, 10, 10, 255));
			}


		}


		

		window.clear(sf::Color::Black);

		myNet.Draw(window);
		
		sf::Texture texture;
		texture.loadFromImage(image);
		sf::Sprite sprite(texture);

		sprite.setScale(10, 10);
		sprite.setPosition(0, 0);
		
		window.draw(sprite);
		window.draw(text);
		

		window.display();
	}
}