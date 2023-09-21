

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
	
	std::vector<std::pair<std::vector<double>, std::vector<double>>> trainingData;
	std::vector<std::vector<double>> testData;


	const int dimensions = 28;

	// empty set
	std::pair<std::vector<double>, std::vector<double>> empty = { std::vector<double>(1, 0), std::vector<double>(28 * 28, 0) };
	trainingData.emplace_back(empty);


	std::vector<unsigned> topology;
	// input layer
	topology.push_back(dimensions * dimensions);

	// hidden layer
	topology.push_back(28);
	topology.push_back(10);
	
	// output layer (10 possible digits)
	topology.push_back(10);


	// read the network
	Network myNet = *Network::Deserialize("network.txt");

	std::vector<double> input;
	std::vector<double> target;
	std::vector<double> results;

	// training per loop
	unsigned trainingPerLoop = 5000;
	
	std::chrono::steady_clock::time_point previous_time;

	// set up window
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


	// app loop
	while (1 == window.isOpen())
	{
		std::chrono::microseconds delta_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - previous_time);

		// poll events
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
		
		float error = 0.0f; // training error

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

				// feed forward
				myNet.feedForward(input);

				// get results
				myNet.getResults(results);

				// back propagate
				myNet.backPropagate(target);



				error += myNet.getRecentAverageError();
			}

			error /= trainingPerLoop;
			

			std::string str = "Error: " + std::to_string(error);
			text.setString(str);

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
			{
				myNet.Serialize();
			}



		}
		else
		{
			// clear user input
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
			// erase
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

			trainingData[0].second = input;
			index = 0;
			

		}

		// save data
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
		{
			myNet.Serialize();
		}

		// print input or current element
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


		
		// draw window
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