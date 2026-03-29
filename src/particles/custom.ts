import { Particle } from "./particle";

export class CustomParticle extends Particle {
  color: [number, number, number];
  readonly groupId: string;

  constructor(x: number, y: number, groupId: string, color: [number, number, number]) {
    super(x, y);
    this.groupId = groupId;
    this.color = color;
  }
}
