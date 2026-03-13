FROM node:20-alpine

WORKDIR /app

# Install dependencies first to leverage Docker layer caching.
COPY package*.json ./
RUN npm install --omit=dev

COPY . .

EXPOSE 8001

CMD ["node", "server.js"]
