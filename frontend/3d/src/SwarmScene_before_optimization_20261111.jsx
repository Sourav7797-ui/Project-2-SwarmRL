import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const CENTER = 100;
const MAX_TRAIL = 90;

export default function SwarmScene({ state }) {
  const mountRef = useRef(null);
  const sceneObjects = useRef(null);
  const histories = useRef({});

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x080d18);
    scene.fog = new THREE.Fog(0x080d18, 260, 520);

    const camera = new THREE.PerspectiveCamera(
      50,
      mount.clientWidth / mount.clientHeight,
      0.1,
      1000
    );
    camera.up.set(0, 0, 1);
    camera.position.set(190, -230, 185);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 15);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.maxPolarAngle = Math.PI * 0.49;
    controls.minDistance = 35;
    controls.maxDistance = 500;

    scene.add(new THREE.HemisphereLight(0x9bbcff, 0x172033, 2.0));

    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(-80, -100, 180);
    sun.castShadow = true;
    scene.add(sun);

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(200, 200),
      new THREE.MeshStandardMaterial({
        color: 0x111d2d,
        roughness: 0.9,
        metalness: 0.1,
      })
    );
    floor.position.set(0, 0, -0.15);
    floor.receiveShadow = true;
    scene.add(floor);

    const grid = new THREE.GridHelper(200, 40, 0x34506b, 0x24374d);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0.02;
    scene.add(grid);

    const axes = new THREE.AxesHelper(18);
    axes.position.set(-100, -100, 0.1);
    scene.add(axes);

    const droneGeometry = new THREE.SphereGeometry(0.65, 12, 10);
    const noseGeometry = new THREE.ConeGeometry(0.38, 1.25, 8);
    noseGeometry.rotateX(Math.PI / 2);
    noseGeometry.translate(0, 0, 0.8);

    const drones = new Map();
    const trails = new Map();

    const obstacleGeometry = new THREE.SphereGeometry(3, 20, 16);
    const obstacleMaterial = new THREE.MeshStandardMaterial({
      color: 0xff754f,
      roughness: 0.65,
      metalness: 0.1,
      transparent: true,
      opacity: 0.72,
    });
    const obstacles = new THREE.Group();
    scene.add(obstacles);

    const coverageGeometry = new THREE.BufferGeometry();
    coverageGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute([], 3)
    );
    const coveragePoints = new THREE.Points(
      coverageGeometry,
      new THREE.PointsMaterial({
        color: 0x36e0b0,
        size: 1.65,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.75,
      })
    );
    scene.add(coveragePoints);

    const boundary = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(200, 200, 40)),
      new THREE.LineBasicMaterial({
        color: 0x4b7399,
        transparent: true,
        opacity: 0.35,
      })
    );
    boundary.position.set(0, 0, 20);
    scene.add(boundary);

    sceneObjects.current = {
      scene,
      camera,
      renderer,
      controls,
      drones,
      trails,
      obstacles,
      obstacleGeometry,
      obstacleMaterial,
      droneGeometry,
      noseGeometry,
      coverageGeometry,
      coveragePoints,
    };

    let frame;
    const animate = () => {
      frame = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const resize = () => {
      const width = mount.clientWidth;
      const height = mount.clientHeight;
      if (!width || !height) return;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(mount);

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      renderer.dispose();
      coverageGeometry.dispose();
      droneGeometry.dispose();
      noseGeometry.dispose();
      obstacleGeometry.dispose();
      obstacleMaterial.dispose();
      mount.removeChild(renderer.domElement);
      sceneObjects.current = null;
    };
  }, []);

  useEffect(() => {
    const objects = sceneObjects.current;
    if (!objects || !state) return;

    const {
      scene,
      drones,
      trails,
      obstacles,
      obstacleGeometry,
      obstacleMaterial,
      droneGeometry,
      noseGeometry,
      coverageGeometry,
    } = objects;

    const agents = state.agents || [];
    const activeIds = new Set(agents.map((agent) => agent.id));

    for (const [id, group] of drones) {
      if (!activeIds.has(id)) {
        scene.remove(group);
        drones.delete(id);
        const trail = trails.get(id);
        if (trail) {
          scene.remove(trail);
          trails.delete(id);
        }
      }
    }

    for (const agent of agents) {
      let group = drones.get(agent.id);

      if (!group) {
        group = new THREE.Group();
        const color = new THREE.Color().setHSL(
          (drones.size * 0.61803398875) % 1,
          0.78,
          0.59
        );

        const body = new THREE.Mesh(
          droneGeometry,
          new THREE.MeshStandardMaterial({
            color,
            emissive: color,
            emissiveIntensity: 0.25,
            roughness: 0.4,
          })
        );
        body.castShadow = true;

        const nose = new THREE.Mesh(
          noseGeometry,
          new THREE.MeshStandardMaterial({
            color: 0xe7f2ff,
            emissive: color,
            emissiveIntensity: 0.35,
          })
        );
        group.add(body, nose);
        scene.add(group);
        drones.set(agent.id, group);

        const trailGeometry = new THREE.BufferGeometry();
        const trail = new THREE.Line(
          trailGeometry,
          new THREE.LineBasicMaterial({
            color,
            transparent: true,
            opacity: 0.45,
          })
        );
        scene.add(trail);
        trails.set(agent.id, trail);
        histories.current[agent.id] = [];
      }

      const [x, y, z] = agent.position;
      group.position.set(x - CENTER, y - CENTER, z);

      const pitch = agent.pitch || 0;
      const yaw = agent.yaw || 0;
      const direction = new THREE.Vector3(
        Math.cos(pitch) * Math.cos(yaw),
        Math.cos(pitch) * Math.sin(yaw),
        Math.sin(pitch)
      ).normalize();

      group.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 0, 1),
        direction
      );

      const body = group.children[0];
      body.material.emissiveIntensity = agent.collision ? 1.2 : 0.25;
      body.material.color.set(agent.collision ? 0xff3434 : body.material.emissive);

      const history = histories.current[agent.id] || [];
      history.push([x - CENTER, y - CENTER, z]);
      if (history.length > MAX_TRAIL) history.shift();
      histories.current[agent.id] = history;

      const trail = trails.get(agent.id);
      trail.geometry.dispose();
      trail.geometry = new THREE.BufferGeometry().setFromPoints(
        history.map(([tx, ty, tz]) => new THREE.Vector3(tx, ty, tz))
      );
    }

    while (obstacles.children.length) {
      const child = obstacles.children[0];
      obstacles.remove(child);
      child.material.dispose();
    }

    for (const [x, y, z] of state.obstacles || []) {
      const mesh = new THREE.Mesh(obstacleGeometry, obstacleMaterial.clone());
      mesh.position.set(x - CENTER, y - CENTER, z);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      obstacles.add(mesh);
    }

    const shape = state.coverage?.shape || [50, 50, 10];
    const covered = state.coverage?.covered_indices || [];
    const positions = new Float32Array(covered.length * 3);
    let offset = 0;

    for (const index of covered) {
      const ix = Math.floor(index / (shape[1] * shape[2]));
      const iy = Math.floor(index / shape[2]) % shape[1];
      const iz = index % shape[2];

      positions[offset++] = ix * 4 + 2 - CENTER;
      positions[offset++] = iy * 4 + 2 - CENTER;
      positions[offset++] = iz * 4 + 2;
    }

    coverageGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(positions, 3)
    );
    coverageGeometry.computeBoundingSphere();
  }, [state]);

  return <div className="scene-mount" ref={mountRef} />;
}
