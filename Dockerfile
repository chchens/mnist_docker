# Use an official Python runtime as a parent image
FROM python:3.5

# Set the working directory to /app
WORKDIR /mydockerbuild

# Copy the current directory contents into the container at /app
ADD . /mydockerbuild

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 5000
EXPOSE 9042

# Define environment variable
ENV NAME MNIST

# Run app.py when the container launches
CMD ["python", "app.py"]
