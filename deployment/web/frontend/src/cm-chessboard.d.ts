declare module "cm-chessboard/src/Chessboard.js" {
  export const COLOR: { readonly white: "w"; readonly black: "b" };
  export const INPUT_EVENT_TYPE: {
    readonly moveInputStarted: "moveInputStarted";
    readonly movingOverSquare: "movingOverSquare";
    readonly validateMoveInput: "validateMoveInput";
    readonly moveInputCanceled: "moveInputCanceled";
    readonly moveInputFinished: "moveInputFinished";
  };

  export interface MoveInputEvent {
    readonly type: string;
    readonly squareFrom: string;
    readonly squareTo?: string;
    readonly legalMove?: boolean;
  }

  export interface ChessboardOptions {
    readonly position: string;
    readonly orientation: "w" | "b";
    readonly assetsUrl: string;
    readonly style: {
      readonly cssClass: string;
      readonly showCoordinates: boolean;
      readonly pieces: { readonly file: string };
      readonly animationDuration: number;
    };
  }

  export class Chessboard {
    public constructor(context: HTMLElement, options: ChessboardOptions);
    public setPosition(fen: string, animated?: boolean): Promise<void>;
    public setOrientation(color: "w" | "b", animated?: boolean): Promise<void>;
    public enableMoveInput(
      handler: (event: MoveInputEvent) => boolean | void,
      color?: "w" | "b",
    ): void;
    public disableMoveInput(): void;
    public destroy(): void;
  }
}

declare module "cm-chessboard/assets/pieces/staunty.svg?url" {
  const url: string;
  export default url;
}
