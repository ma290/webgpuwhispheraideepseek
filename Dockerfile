# Use a lightweight Nginx image to serve static files
FROM nginx:alpine

# Copy the custom Nginx config to listen on port 8000
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the static web files to the Nginx html directory
COPY index.html /usr/share/nginx/html/
COPY style.css /usr/share/nginx/html/
COPY app.js /usr/share/nginx/html/

# Expose port 8000 for the web server
EXPOSE 8000

# Start Nginx server
CMD ["nginx", "-g", "daemon off;"]
