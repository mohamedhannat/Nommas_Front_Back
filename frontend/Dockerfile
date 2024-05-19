# Use an official node runtime as a parent image
FROM node:14-alpine

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the current directory contents into the container
COPY . .

# Build the app
RUN npm run build

# Serve the app with a simple static server
RUN npm install -g serve

# Make port 3000 available to the world outside this container
EXPOSE 3000

# Run serve when the container launches
CMD ["serve", "-s", "build"]
