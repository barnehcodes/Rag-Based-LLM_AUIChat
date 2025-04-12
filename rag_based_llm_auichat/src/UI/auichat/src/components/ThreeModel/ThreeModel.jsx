import { useEffect, useRef, useState } from 'react';
import { Box, useTheme } from '@mui/material';
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { animate, createScope, createSpring } from 'animejs';
import modelObj from '../../assets/obj/symbolic_lightbulb_wi_0411185454_texture.obj?url';
import modelMtl from '../../assets/obj/symbolic_lightbulb_wi_0411185454_texture.mtl?url';
import modelTexture from '../../assets/obj/symbolic_lightbulb_wi_0411185454_texture.png';

const ThreeModel = () => {
  const themeContext = useTheme();
  const isDark = themeContext.palette.mode === 'dark';
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const modelRef = useRef(null);
  const controlsRef = useRef(null);
  const animationScope = useRef(null);
  
  const [loaded, setLoaded] = useState(false);

  // Add refs for new light elements
  const bulbLightRef = useRef(null);
  const orbitRingRef = useRef(null);
  const glowMeshRef = useRef(null);

  useEffect(() => {
    // Initialize Three.js scene
    const initThree = () => {
      if (!containerRef.current) return;
      
      // Clean up any existing canvas elements first to prevent duplicates
      while (containerRef.current.firstChild) {
        containerRef.current.removeChild(containerRef.current.firstChild);
      }
      
      // Scene setup
      const scene = new THREE.Scene();
      sceneRef.current = scene;
      
      // Camera setup
      const aspectRatio = containerRef.current.clientWidth / containerRef.current.clientHeight;
      const camera = new THREE.PerspectiveCamera(45, aspectRatio, 0.1, 1000);
      camera.position.set(0, 0, 5);
      cameraRef.current = camera;
      
      // Renderer setup with improved settings
      const renderer = new THREE.WebGLRenderer({ 
        antialias: true, 
        alpha: true,
        powerPreference: "high-performance" 
      });
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
      renderer.setClearColor(0x000000, 0);
      renderer.setPixelRatio(window.devicePixelRatio);
      
      // Enable shadow mapping for realistic shadows
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      
      containerRef.current.appendChild(renderer.domElement);
      rendererRef.current = renderer;
      
      // Add orbit ring to visualize the orbit path
      const orbitGeometry = new THREE.TorusGeometry(3.5, 0.03, 16, 100);
      const orbitMaterial = new THREE.MeshBasicMaterial({ 
        color: isDark ? 0x6a1b9a : 0x9c27b0,
        transparent: true,
        opacity: 0.3
      });
      const orbitRing = new THREE.Mesh(orbitGeometry, orbitMaterial);
      orbitRing.rotation.x = Math.PI / 2; // Tilt to make it horizontal
      scene.add(orbitRing);
      orbitRingRef.current = orbitRing;
      
      // Enhanced lighting setup
      const ambientLight = new THREE.AmbientLight(isDark ? 0x333333 : 0x666666, 0.5);
      scene.add(ambientLight);
      
      const directionalLight1 = new THREE.DirectionalLight(isDark ? 0xccccff : 0xffffff, 0.8);
      directionalLight1.position.set(1, 1, 1);
      directionalLight1.castShadow = true;
      directionalLight1.shadow.mapSize.width = 1024;
      directionalLight1.shadow.mapSize.height = 1024;
      scene.add(directionalLight1);
      
      const directionalLight2 = new THREE.DirectionalLight(isDark ? 0xccccff : 0xffffff, 0.3);
      directionalLight2.position.set(-1, -1, -1);
      scene.add(directionalLight2);
      
      // Bulb light - this will be the light emanating from the bulb
      const bulbLight = new THREE.PointLight(0xffffcc, 3.5, 15, 2); // Increased intensity and distance
      bulbLight.position.set(0, 0, 0);
      bulbLight.castShadow = true;
      scene.add(bulbLight);
      bulbLightRef.current = bulbLight;
      
      // Add a glow effect around the bulb
      const glowGeometry = new THREE.SphereGeometry(0.8, 32, 32); // Larger glow sphere
      const glowMaterial = new THREE.MeshBasicMaterial({
        color: 0xffffdd,
        transparent: true,
        opacity: 0.7, // Increased opacity
        side: THREE.BackSide
      });
      const glowMesh = new THREE.Mesh(glowGeometry, glowMaterial);
      scene.add(glowMesh);
      glowMeshRef.current = glowMesh;
      
      // OrbitControls setup with improved settings
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.rotateSpeed = 0.5;
      controls.minDistance = 3;
      controls.maxDistance = 10;
      controls.autoRotate = true;  // Enable autorotation for dynamic effect
      controls.autoRotateSpeed = 0.5; // Slow rotation
      controlsRef.current = controls;
      
      // Load 3D model
      loadModel();
      
      // Animation loop with enhanced effects
      const animate = () => {
        requestAnimationFrame(animate);
        
        if (controlsRef.current) {
          controlsRef.current.update();
        }
        
        if (modelRef.current) {
          // Add subtle breathing animation to model
          const time = Date.now() * 0.001;
          const scale = 1 + Math.sin(time) * 0.03;
          modelRef.current.scale.set(scale, scale, scale);
          
          // Make the bulb light pulse
          if (bulbLightRef.current) {
            bulbLightRef.current.intensity = 3.0 + Math.sin(time * 2) * 0.8; // Increased base intensity and pulse range
            
            // Update light position to match the bulb
            const bulbPosition = new THREE.Vector3();
            bulbPosition.setFromMatrixPosition(modelRef.current.matrixWorld);
            bulbLightRef.current.position.copy(bulbPosition);
            
            // Update glow effect position
            if (glowMeshRef.current) {
              glowMeshRef.current.position.copy(bulbPosition);
              const glowScale = 2.0 + Math.sin(time * 2) * 0.3; // Larger base scale and animation range
              glowMeshRef.current.scale.set(glowScale, glowScale, glowScale);
              glowMeshRef.current.material.opacity = 0.6 + Math.sin(time * 2) * 0.2; // Higher base opacity
            }
          }
        }
        
        // Make orbit ring pulse subtly
        if (orbitRingRef.current) {
          const time = Date.now() * 0.001;
          orbitRingRef.current.material.opacity = 0.2 + Math.sin(time) * 0.1;
        }
        
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      };
      
      animate();
      
      // Handle window resize
      const handleResize = () => {
        if (!containerRef.current || !cameraRef.current || !rendererRef.current) return;
        
        const width = containerRef.current.clientWidth;
        const height = containerRef.current.clientHeight;
        
        cameraRef.current.aspect = width / height;
        cameraRef.current.updateProjectionMatrix();
        
        rendererRef.current.setSize(width, height);
      };
      
      window.addEventListener('resize', handleResize);
      
      return () => {
        window.removeEventListener('resize', handleResize);
        if (rendererRef.current && containerRef.current) {
          containerRef.current.removeChild(rendererRef.current.domElement);
          rendererRef.current.dispose();
        }
      };
    };
    
    const loadModel = () => {
      const textureLoader = new THREE.TextureLoader();
      const texture = textureLoader.load(modelTexture);
      
      const mtlLoader = new MTLLoader();
      mtlLoader.load(modelMtl, (materials) => {
        materials.preload();
        
        // Enhance material properties for better visual appearance
        Object.keys(materials.materials).forEach(key => {
          const material = materials.materials[key];
          
          // Make metal parts shinier
          if (key.includes('metal') || key.includes('base')) {
            material.shininess = 100;
            material.specular.set(0xffffff);
          }
          
          // Make glass parts of the bulb transparent and refractive
          if (key.includes('glass') || key.includes('bulb')) {
            material.transparent = true;
            material.opacity = 0.85;
            material.shininess = 100;
            material.specular.set(0xffffff);
            // Add emissive glow to the bulb
            material.emissive.set(0xffffcc);
            material.emissiveIntensity = 0.9; // Increased intensity
          }
        });
        
        const objLoader = new OBJLoader();
        objLoader.setMaterials(materials);
        objLoader.load(modelObj, (object) => {
          // Center and scale the model
          const box = new THREE.Box3().setFromObject(object);
          const center = box.getCenter(new THREE.Vector3());
          const size = box.getSize(new THREE.Vector3());
          
          const maxDim = Math.max(size.x, size.y, size.z);
          const scale = 2.5 / maxDim;
          object.scale.set(scale, scale, scale);
          
          object.position.sub(center.multiplyScalar(scale));
          object.position.y -= 0.5; // Adjust vertical position
          
          // Enable shadows for the object
          object.traverse(function(child) {
            if (child instanceof THREE.Mesh) {
              child.castShadow = true;
              child.receiveShadow = true;
              
              // If the mesh is part of the glass bulb, make it emit light
              if (child.material.name && 
                 (child.material.name.includes('glass') || 
                  child.material.name.includes('bulb'))) {
                child.material.emissive = new THREE.Color(0xffffdd);
                child.material.emissiveIntensity = 1.2; // Increased intensity
              }
            }
          });
          
          // Rotate model to face camera
          object.rotation.y = Math.PI;
          
          sceneRef.current.add(object);
          modelRef.current = object;
          
          // Position the glow and light effects
          if (bulbLightRef.current && glowMeshRef.current) {
            const bulbPosition = new THREE.Vector3();
            bulbPosition.setFromMatrixPosition(object.matrixWorld);
            bulbLightRef.current.position.copy(bulbPosition);
            glowMeshRef.current.position.copy(bulbPosition);
          }
          
          // Use a simple animation for the model loading instead of anime.js
          // This avoids the TypeError we were getting
          const startY = object.position.y;
          const targetY = startY + 0.2;
          let startTime = null;
          const duration = 1500;
          
          function animateModelPosition(timestamp) {
            if (!startTime) startTime = timestamp;
            const elapsed = timestamp - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Use an easing function similar to easeOutElastic
            const easeOut = function(t) {
              return t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
            };
            
            const y = startY + (targetY - startY) * easeOut(progress);
            object.position.y = y;
            
            if (progress < 1) {
              requestAnimationFrame(animateModelPosition);
            } else {
              setLoaded(true);
            }
          }
          
          requestAnimationFrame(animateModelPosition);
          
          // Setup animation scope for scroll interactions
          if (containerRef.current) {
            animationScope.current = createScope({ root: containerRef }).add(scope => {
              // We'll handle scroll animations without using anime.js directly on Three.js objects
              scope.add('animateOnScroll', (scrollProgress) => {
                // Directly modify the Three.js objects instead of using anime.js
                if (modelRef.current) {
                  modelRef.current.rotation.y = Math.PI + scrollProgress * Math.PI * 2;
                }
                
                if (cameraRef.current) {
                  cameraRef.current.position.z = 5 - scrollProgress * 1.5;
                  cameraRef.current.updateProjectionMatrix();
                }
              });
            });
          }
        });
      });
    };
    
    initThree();
    
    return () => {
      if (animationScope.current) {
        animationScope.current.revert();
      }
      
      // Ensure proper cleanup of Three.js resources
      if (rendererRef.current) {
        rendererRef.current.dispose();
        if (rendererRef.current.domElement && rendererRef.current.domElement.parentNode) {
          rendererRef.current.domElement.parentNode.removeChild(rendererRef.current.domElement);
        }
      }
      
      // Clear any remaining children from the container
      if (containerRef.current) {
        while (containerRef.current.firstChild) {
          containerRef.current.removeChild(containerRef.current.firstChild);
        }
      }
    };
  }, [isDark]);
  
  // Handle scroll interaction
  useEffect(() => {
    const handleScroll = () => {
      if (!loaded || !animationScope.current || !animationScope.current.methods.animateOnScroll) return;
      
      const scrollY = window.scrollY;
      const scrollProgress = Math.min(scrollY / 1000, 1); // Limit effect to first 1000px of scroll
      
      // Use the method created in the animation scope
      animationScope.current.methods.animateOnScroll(scrollProgress);
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [loaded]);
  
  return (
    <Box
      ref={containerRef}
      sx={{
        width: '100%',
        height: '40vh', // Reduced height to fit better between title and chat
        minHeight: '250px', // Reduced minimum height
        position: 'relative',
        marginBottom: '2rem', // Reduced bottom margin to move chat closer
        marginTop: '1rem',
        zIndex: 2, // Ensure proper z-index for layering
        overflow: 'hidden', // Prevent any overflow issues
        '& canvas': {
          display: 'block !important', // Force proper display
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100% !important',
          height: '100% !important',
          zIndex: 2
        }
      }}
    />
  );
};

export default ThreeModel;