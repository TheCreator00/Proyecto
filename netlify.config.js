// Configuración optimizada para Netlify
module.exports = {
  // Optimizaciones de construcción
  build: {
    processing: {
      // Optimización de imágenes
      images: {
        compress: true,
        quality: 85,
        sizes: [400, 800, 1200],
        formats: ['webp', 'avif']
      },
      // Optimización de CSS
      css: {
        minify: true,
        purge: true
      },
      // Optimización de JavaScript
      js: {
        minify: true,
        bundle: true
      }
    },
    // Cache y performance
    cache: {
      edge: true,
      browser: {
        include: ['*.html', '*.css', '*.js', '*.json', '*.svg']
      }
    }
  },

  // Headers de seguridad y rendimiento
  headers: {
    '/*': [
      {
        key: 'X-Frame-Options',
        value: 'DENY'
      },
      {
        key: 'X-Content-Type-Options',
        value: 'nosniff'
      },
      {
        key: 'Referrer-Policy',
        value: 'strict-origin-when-cross-origin'
      },
      {
        key: 'Cache-Control',
        value: 'public, max-age=31536000'
      }
    ]
  },

  // Configuración de funciones serverless
  functions: {
    directory: 'dist/functions',
    included_files: ['*.py', 'config/*'],
    external_node_modules: ['tensorflow', 'torch'],
    node_bundler: 'esbuild'
  },

  // Redirecciones y reescrituras
  redirects: [
    {
      from: '/*',
      to: '/index.html',
      status: 200,
      force: false
    }
  ],

  // Plugins recomendados
  plugins: [
    '@netlify/plugin-lighthouse',
    '@netlify/plugin-sitemap',
    '@netlify/plugin-caching-headers'
  ]
}