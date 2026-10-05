FROM node:22-alpine

WORKDIR /app

COPY package.json package-lock.json ./

RUN npm ci

COPY . .

# Vite exposes on 5173 by default
EXPOSE 5173

# Need host mapping for Docker port forwarding to work with Vite
CMD ["npm", "run", "dev", "--", "--host"]
