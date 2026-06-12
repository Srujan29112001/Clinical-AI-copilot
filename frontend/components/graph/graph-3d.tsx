"use client";
import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { GraphNode, KnowledgeGraph } from "@/lib/types";
import { GROUP_COLORS } from "@/lib/utils";

/**
 * 3D interactive knowledge graph — raw three.js (no drei, so it builds clean on
 * React 19 / Next 15). Spheres = nodes (coloured by ontology type), lines =
 * relationships. A 3D force layout is pre-computed, then the scene auto-rotates
 * with OrbitControls; raycasting drives hover/click → onSelect.
 */
export function Graph3D({
  graph, activeGroups, onSelect,
}: {
  graph: KnowledgeGraph;
  activeGroups: Set<string>;
  onSelect: (n: GraphNode | null) => void;
}) {
  const mountRef = useRef<HTMLDivElement>(null);
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;
  const activeRef = useRef(activeGroups);
  activeRef.current = activeGroups;

  useEffect(() => {
    const mount = mountRef.current!;
    const W = mount.clientWidth, H = mount.clientHeight;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, W / H, 0.1, 4000);
    camera.position.set(0, 0, 620);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(W, H);
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.9));
    const pt = new THREE.PointLight(0x22d3ee, 1.2); pt.position.set(200, 200, 300); scene.add(pt);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.5;

    // ── 3D force layout (precomputed) ──
    const idx = new Map<string, number>();
    graph.nodes.forEach((n, i) => idx.set(n.id, i));
    const N = graph.nodes.length;
    const pos = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      const r = 200 + Math.random() * 120;
      const th = Math.random() * Math.PI * 2, ph = Math.acos(2 * Math.random() - 1);
      pos[i * 3] = r * Math.sin(ph) * Math.cos(th);
      pos[i * 3 + 1] = r * Math.sin(ph) * Math.sin(th);
      pos[i * 3 + 2] = r * Math.cos(ph);
    }
    const edges = graph.edges
      .map((e) => [idx.get(e.source), idx.get(e.target)])
      .filter(([a, b]) => a !== undefined && b !== undefined) as [number, number][];
    const deg = new Array(N).fill(0);
    edges.forEach(([a, b]) => { deg[a]++; deg[b]++; });

    for (let iter = 0; iter < 120; iter++) {
      const disp = new Float32Array(N * 3);
      for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
          let dx = pos[i * 3] - pos[j * 3], dy = pos[i * 3 + 1] - pos[j * 3 + 1], dz = pos[i * 3 + 2] - pos[j * 3 + 2];
          let d2 = dx * dx + dy * dy + dz * dz || 1;
          const f = 90000 / d2;
          const d = Math.sqrt(d2); dx /= d; dy /= d; dz /= d;
          disp[i * 3] += dx * f; disp[i * 3 + 1] += dy * f; disp[i * 3 + 2] += dz * f;
          disp[j * 3] -= dx * f; disp[j * 3 + 1] -= dy * f; disp[j * 3 + 2] -= dz * f;
        }
      }
      for (const [a, b] of edges) {
        let dx = pos[b * 3] - pos[a * 3], dy = pos[b * 3 + 1] - pos[a * 3 + 1], dz = pos[b * 3 + 2] - pos[a * 3 + 2];
        const d = Math.hypot(dx, dy, dz) || 1;
        const f = (d - 80) * 0.04;
        dx /= d; dy /= d; dz /= d;
        disp[a * 3] += dx * f; disp[a * 3 + 1] += dy * f; disp[a * 3 + 2] += dz * f;
        disp[b * 3] -= dx * f; disp[b * 3 + 1] -= dy * f; disp[b * 3 + 2] -= dz * f;
      }
      const cool = 1 - iter / 120;
      for (let i = 0; i < N * 3; i++) {
        pos[i] += Math.max(-8, Math.min(8, disp[i] * 0.02)) * cool - pos[i] * 0.001 * cool;
      }
    }

    // ── nodes (instanced spheres) ──
    const geo = new THREE.SphereGeometry(1, 16, 16);
    // NOTE: per-instance colours come from setColorAt() → instanceColor buffer.
    // Do NOT set vertexColors:true (that reads a non-existent geometry colour attr → black).
    const mat = new THREE.MeshBasicMaterial();  // unlit white, multiplied by instanceColor
    const mesh = new THREE.InstancedMesh(geo, mat, N);
    const dummy = new THREE.Object3D();
    const color = new THREE.Color();
    for (let i = 0; i < N; i++) {
      dummy.position.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
      const s = 4 + Math.min(deg[i], 10) * 1.4;
      dummy.scale.setScalar(s);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      color.set(GROUP_COLORS[graph.nodes[i].group] || "#8ea0c4");
      mesh.setColorAt(i, color);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    scene.add(mesh);

    // ── edges ──
    const lineGeo = new THREE.BufferGeometry();
    const linePos = new Float32Array(edges.length * 6);
    edges.forEach(([a, b], k) => {
      linePos.set([pos[a * 3], pos[a * 3 + 1], pos[a * 3 + 2], pos[b * 3], pos[b * 3 + 1], pos[b * 3 + 2]], k * 6);
    });
    lineGeo.setAttribute("position", new THREE.BufferAttribute(linePos, 3));
    const lines = new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({ color: 0x2a3a5c, transparent: true, opacity: 0.35 }));
    scene.add(lines);

    // ── label sprite (for hover) ──
    const labelDiv = document.createElement("div");
    labelDiv.style.cssText = "position:absolute;pointer-events:none;background:rgba(10,15,26,.92);border:1px solid #1e2a44;color:#eaf0ff;font:11px Inter,sans-serif;padding:3px 7px;border-radius:6px;transform:translate(-50%,-130%);display:none;white-space:nowrap;z-index:10";
    mount.appendChild(labelDiv);

    // ── raycasting ──
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let hovered = -1;
    const onMove = (e: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const hit = raycaster.intersectObject(mesh);
      const visibleHit = hit.find((h) => activeRef.current.has(graph.nodes[h.instanceId!].group));
      if (visibleHit) {
        hovered = visibleHit.instanceId!;
        const n = graph.nodes[hovered];
        labelDiv.textContent = n.label;
        labelDiv.style.display = "block";
        labelDiv.style.left = e.clientX - rect.left + "px";
        labelDiv.style.top = e.clientY - rect.top + "px";
        renderer.domElement.style.cursor = "pointer";
      } else {
        hovered = -1; labelDiv.style.display = "none"; renderer.domElement.style.cursor = "grab";
      }
    };
    const onClick = () => { if (hovered >= 0) onSelectRef.current(graph.nodes[hovered]); };
    renderer.domElement.addEventListener("mousemove", onMove);
    renderer.domElement.addEventListener("click", onClick);

    // ── visibility update loop reads activeRef ──
    let raf = 0;
    const animate = () => {
      // dim instances whose group is filtered out
      for (let i = 0; i < N; i++) {
        const vis = activeRef.current.has(graph.nodes[i].group);
        color.set(vis ? (GROUP_COLORS[graph.nodes[i].group] || "#8ea0c4") : "#0e1524");
        mesh.setColorAt(i, color);
      }
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    };
    animate();

    const onResize = () => {
      const w = mount.clientWidth, h = mount.clientHeight;
      camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h);
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      renderer.domElement.removeEventListener("mousemove", onMove);
      renderer.domElement.removeEventListener("click", onClick);
      controls.dispose();
      renderer.dispose();
      geo.dispose(); mat.dispose(); lineGeo.dispose();
      mount.removeChild(renderer.domElement);
      mount.removeChild(labelDiv);
    };
  }, [graph]);

  return <div ref={mountRef} className="h-full w-full" />;
}
