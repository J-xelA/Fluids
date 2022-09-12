///////////////////////////////////////////////////////////////////////////////////////////
// Download Simple and Fast Multimedia Library for all GUI components:                   //
// https://www.sfml-dev.org/download/sfml/2.5.1/                                         //
//                                                                                       //
// Configure SFML for Visual Studio: https://www.sfml-dev.org/tutorials/2.5/start-vc.php //
///////////////////////////////////////////////////////////////////////////////////////////

#include <SFML/Graphics.hpp>
#include "SliderSFML.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <string>

using std::vector;
using std::to_string;
using std::swap;

// Average FPS for same data size
// 720p  1280x720:  72 fps  11.9  ms
// 1080p 1920x1080: 34 fps  25.4  ms
// 2K    2560x1440: 17 fps  51.8  ms
// 4K    3840x2160: 8  fps  106.1 ms

// Average FPS for 16:9 reduced data size
// 720p  1280x720:  197 fps   4.2  ms
// 1080p 1920x1080: 66  fps   11.5 ms
// 2K    2560x1440: 57  fps   14.7 ms
// 4K    3840x2160: 22  fps   41.4 ms

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720
#define CUDA_X WINDOW_WIDTH/1.7
#define CUDA_Y WINDOW_HEIGHT/1.7

// Fluid computation
void compute(uint8_t* result, int x1, int y1, int x2, int y2, float dt, bool mousePressed);
// Data allocation
void cudaInit(size_t, size_t);
// Data deletion
void cudaExit();

int main() {
	// Allocate resources
	cudaInit(CUDA_X, CUDA_Y);

	// Create graphics window
	sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Boof");

	//window.setFramerateLimit(60); // Limit frame rate
	SliderSFML radiusSlider(100, 100);
	SliderSFML vorticitySlider(100, 200);

	radiusSlider.create(10, 300);
	vorticitySlider.create(0, 100);

	radiusSlider.setSliderValue(250);
	vorticitySlider.setSliderValue(50);

	//int radius = 250; // Radius of the emitter
	//float vorticity = 50.0f; // Controls rotation (Higher is more chaotic)
	//float velocityDiffusion = 0.75f; // Number of trails/spikes (Higher adds more)
	//float colorDiffusion = 0.5f; // Color resolution (Higher adds less resolution)
	//float densityDiffusion = 0.025f; // Duration (Higher dissipates slower)
	//float forceScale = 2500.0f; // Emission force
	//float pressure = 1.25f;

	// 2D vector holding cursor position for two events (move & click)
	sf::Vector2<int> mouseButton(0, 0), mouseMove(0, 0);

	sf::Texture texture; // Create a texture (Image on the device)
	sf::Sprite sprite; // Create a sprite (Drawable representation of a texture)

	vector<sf::Uint8> pixels(CUDA_X * CUDA_Y * 4); // Array of pixel data
	texture.create(CUDA_X, CUDA_Y); // Texture with data width and height

	sf::Clock clock; // Used for fps counter
	float start = 0; // Starting time (fps)

	bool mousePressed = false; // Current state of the mouse

	// Loop as long as the graphics window is open
	while (window.isOpen()) {
		float end = clock.restart().asSeconds(); // Ending time (fps)
		float fps = 1.0f / end; // Compute fps
		start = end; // Reset starting time
		// If fps is positive
		if (fps > 0)
			window.setTitle(to_string(int(fps)) + " FPS"); // Print the fps as the title

		sf::Event event; // Create an event

		// Pop the event on top of the event queue, if any, process it
		while (window.pollEvent(event)) {
			// Close the window when you click 'X'
			if (event.type == sf::Event::Closed)
				window.close();
			// Look for LMB press and change its state
			if (event.type == sf::Event::MouseButtonPressed) {
				if (event.mouseButton.button == sf::Mouse::Button::Left) {
					mouseButton = { event.mouseButton.x /= 3, event.mouseButton.y /= 3 };
					mousePressed = true;
				}//if
			}//if
			// Look for mouse movements
			if (event.type == sf::Event::MouseMoved) {
				swap(mouseButton, mouseMove);
				mouseMove = { event.mouseMove.x /= 3, event.mouseMove.y /= 3 };
			}//if
			// Change the state of the mouse button
			if (event.type == sf::Event::MouseButtonReleased)
				mousePressed = false;
		}//while

		float dt = 0.0125f; // Time scale
		// Compute all necessary data for the pixel array
		compute(pixels.data(), mouseButton.x, mouseButton.y, mouseMove.x, mouseMove.y, dt, mousePressed);

		// Add the pixel data to the texture
		texture.update(pixels.data());

		// Set the sprite as the texture and scale it
		sprite.setTexture(texture);
		sprite.setScale({ 3, 3 });

		// Draw the sprite to the window and display it
		window.draw(sprite);

		//radiusSlider.draw(window);
		//vorticitySlider.draw(window);

		window.display();
	}//while

	// Free resources
	cudaExit();

	return 0;
}//main()