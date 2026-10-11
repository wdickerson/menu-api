
## Version 1

The API accepts between 1 and 10 images in its single POST endpoint, and makes a call
to the OpenAI API to transcribe the menu image into the JSON format needed by menu-frontend.

menu-api needs to know an OpenAI API key to make the request, which can be provided by a human developer.

The usual successful response is JSON including the menu data and the restaurant name. If the restaurant name cannot be determined, then a descriptive placeholder is given, like "Burger Place", "Coffee Shop", "Indian Restaurant", etc.

The images may be of a printed menu, a large typewritten menu behind a counter (like at a counter-serve restaurant), a poster-sized menu (like at a food truck), a large handwritten menu (like the draft beer list at a beer hall), etc.

Failure responses may include:
Text in image is not legible
No menu found in image
OpenAI API is not funded
Other OpenAI failure
Any other errors that make sense to include

A special "dummy" flag is considered on the post input which tells menu-api to return a stored, dummy menu instead of making a call to OpenAI. This is so that the frontend can be developed without always having to call OpenAI.


