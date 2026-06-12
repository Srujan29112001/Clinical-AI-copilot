import { Hero } from "@/components/landing/hero";
import {
  Stats, AgentRoster, Pipeline, HybridInference, TechStack, CTA,
} from "@/components/landing/sections";

export default function HomePage() {
  return (
    <>
      <Hero />
      <Stats />
      <AgentRoster />
      <Pipeline />
      <HybridInference />
      <TechStack />
      <CTA />
    </>
  );
}
