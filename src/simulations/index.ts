import type { Simulation } from "../types";
import { RandomDots } from "./basic-particles";

/** Registry of all available simulations. Add new ones here. */
export const simulations: Simulation[] = [new RandomDots()];
