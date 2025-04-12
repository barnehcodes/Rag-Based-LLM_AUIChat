Project Goal: Create a modern, interactive website UI for an AUI Chatbot. The site should feature a 3D model, dynamic background, the chatbot interface itself, and showcase the technologies used.

Core Technologies:

    Framework/Library: React.js
    UI Components: Material-UI (MUI)
    3D Rendering: Three.js
    Animation: Anime.js

Key Features & Sections:

    Header / Top Menu Bar:
        Include the website logo (auichat-high-resolution-logo.png).
        Display icons linking to external sites: GitHub, LinkedIn, and a "Docs" page (provide placeholder links if final URLs aren't ready). Use appropriate icons (e.g., from MUI Icons or FontAwesome).
        Should likely be fixed or sticky for easy access.
        Integrate the Light/Dark Mode toggle switch here.

    Interactive 3D Model Section:
        Load and display the symbolic_lightbulb_wi_0411113250_texture.obj 3D model using Three.js.
        Apply associated textures and materials (.mtl, .png) to the model.
        Implement user interaction: Allow users to rotate/move the lightbulb model using mouse drag (e.g., using Three.js OrbitControls).
        Implement scroll-based interaction: As the user scrolls down the page, the orbit/view of the 3D model should change subtly (e.g., camera zooms slightly, rotates around the Y-axis). Use Anime.js triggered by scroll events.

    Interactive Background:
        Create a dynamic background with a "cloud-ish," "spacy" feel.
        Use a gradient color scheme: Dark purple, blue, and black.
        Implement a parallax effect: As the user scrolls vertically down the page, the background should appear to shift horizontally (to the right).

    Chatbot Interface Section:
        Design a dedicated section for the chatbot interaction.
        Use MUI components for styling.
        Include a standard chat message display area (scrollable).
        Include a text input field for user queries.
        Include a "Send" button.
        Display metrics near the chat interface: "Inference Time," and potentially other relevant metrics (use placeholder values initially).
        Enclose this entire section within a styled container (e.g., MUI Card or Box) that has a noticeable shadow for visual separation.

    Technology Showcase Section:
        Visually represent the core technologies used (e.g., React, MUI, Three.js, Anime.js, Python/Backend Tech if applicable).
        Display these as interactive "bubbles" or "balls".
        Add subtle animations using Anime.js (e.g., gentle floating, reaction on hover/click).

    Light/Dark Mode:
        Implement a theme toggle (likely in the Header).
        Use MUI's theming capabilities to define distinct color palettes, background colors, and component styles for both light and dark modes. Ensure all components adapt correctly.

Asset Information:

    All required assets are located in the src/assets/ directory relative to the source code root.
    Assets available:
        auichat-high-resolution-logo.png (Website Logo)
        symbolic_lightbulb_wi_0411113250_texture.obj (3D Model)
        symbolic_lightbulb_wi_0411113250_texture.mtl (Material File for 3D Model)
        symbolic_lightbulb_wi_0411113250_texture.png (Texture for 3D Model)

General Requirements:

    Responsiveness: The layout should adapt reasonably well to different screen sizes (desktop, tablet). Use MUI's grid system and responsive utilities.
    Interactivity: Ensure smooth animations and interactions as specified for the 3D model, background, and tech bubbles.
    Code Structure: Generate well-structured React components.

Output: Provide the necessary React component files (JSX/TSX), CSS/styling (preferably using MUI's styling solutions like sx prop or styled-components), and JavaScript logic for interactions, Three.js setup, and Anime.js animations.
