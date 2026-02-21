Feature: EffectRule System
  As an artist who experiments with layered visual effects,
  I want to define composable, chainable processing rules,
  So that I can mix and match effects and have them carry state smoothly across video frames.

  Background:
    Given the EffectRule system is installed and configured
    And at least one input image or video frame is available

  # --- Single Rule Execution ---

  Scenario: Apply a single stateless EffectRule to an image
    Given an input image "portrait.jpg"
    And a rule chain configured as [EdgeDetect] with detail_level 5
    When I run the rule chain
    Then the output is a single processed image
    And the image contains edge detection output at detail level 5

  Scenario: Apply a single stateful EffectRule to a video frame sequence
    Given a sequence of 30 video frames
    And a rule chain configured as [HalfToneRandomization] with temporal_smoothing 0.8
    When I run the rule chain on all 30 frames
    Then each frame produces a halftone output
    And the halftone pattern evolves smoothly from frame 1 to frame 30

  Scenario: Apply a rule with custom configuration
    Given an input image "portrait.jpg"
    And a rule chain configured as [EdgeDetect] with algorithm "canny" and low_threshold 30 and high_threshold 100
    When I run the rule chain
    Then the output uses Canny edge detection with the specified thresholds
    And the output differs from running with default thresholds

  # --- Chain Composition ---

  Scenario: Chain two rules in sequence
    Given an input image "portrait.jpg"
    And a rule chain configured as [EdgeDetect, ThresholdBinarize]
    When I run the rule chain
    Then EdgeDetect processes the image first
    And ThresholdBinarize receives the output of EdgeDetect as its input
    And the final output is a binarized edge map

  Scenario: Chain three rules in sequence
    Given an input image "portrait.jpg"
    And a rule chain configured as [SilhouetteExtract, EdgeDetect, DepthComposite]
    When I run the rule chain
    Then each rule receives the output of the preceding rule
    And the final output combines silhouette isolation, edge detection, and depth compositing

  Scenario: Chain order affects output
    Given an input image "portrait.jpg"
    And rule chain A configured as [EdgeDetect, ThresholdBinarize]
    And rule chain B configured as [ThresholdBinarize, EdgeDetect]
    When I run both rule chains on the same input
    Then the outputs of chain A and chain B are visually different
    And both outputs are valid processed images

  Scenario: Long chain with many rules
    Given an input image "portrait.jpg"
    And a rule chain with 6 rules: [SilhouetteExtract, EdgeDetect, DepthMap, ContourOverlay, HalfToneRandomization, TemporalSmooth]
    When I run the rule chain
    Then all 6 rules execute in order
    And the final output reflects the cumulative effect of all rules
    And no intermediate data is lost between rules

  # --- Rule Swapping and Modularity ---

  Scenario: Swap one rule for another in a chain
    Given a rule chain configured as [EdgeDetect(algorithm="canny"), DepthComposite]
    When I replace EdgeDetect with EdgeDetect(algorithm="hed") in the chain
    And I run the updated rule chain
    Then the chain executes with HED edge detection instead of Canny
    And DepthComposite is unaffected by the swap
    And the output is valid

  Scenario: Remove a rule from a chain
    Given a rule chain configured as [SilhouetteExtract, EdgeDetect, DepthComposite]
    When I remove DepthComposite from the chain
    And I run the updated rule chain
    Then only SilhouetteExtract and EdgeDetect execute
    And the output does not include depth compositing

  Scenario: Add a rule to an existing chain
    Given a rule chain configured as [EdgeDetect]
    When I append HalfToneRandomization to the chain
    And I run the updated rule chain
    Then EdgeDetect runs first, followed by HalfToneRandomization
    And the output includes both edge detection and halftone effects

  Scenario: Reorder rules in a chain
    Given a rule chain configured as [DepthComposite, EdgeDetect, SilhouetteExtract]
    When I reorder the chain to [SilhouetteExtract, EdgeDetect, DepthComposite]
    And I run the reordered chain
    Then the rules execute in the new order
    And the output reflects the reordered processing sequence

  # --- Inter-Frame State Management ---

  Scenario: Stateful rule maintains state across frames
    Given a sequence of 10 video frames
    And a rule chain configured as [HalfToneRandomization] with temporal_smoothing 0.8
    When I process frames 1 through 10 in order
    Then the rule's internal state is updated after each frame
    And frame 10 output reflects accumulated state from all previous frames

  Scenario: Stateless rule produces identical output regardless of frame order
    Given a sequence of 10 video frames
    And a rule chain configured as [EdgeDetect] with detail_level 5
    When I process frame 5 in isolation
    And I process frame 5 as part of the full sequence
    Then both outputs for frame 5 are identical

  Scenario: Reset state clears inter-frame accumulation
    Given a sequence of 10 video frames processed through a stateful rule
    When I reset the rule's state
    And I process frame 11
    Then frame 11 output is as if it were the first frame processed
    And no state from frames 1-10 influences frame 11

  Scenario: Multiple stateful rules in a chain maintain independent state
    Given a sequence of 10 video frames
    And a rule chain configured as [HalfToneRandomization, TemporalSmooth]
    When I process all 10 frames
    Then HalfToneRandomization maintains its own internal state
    And TemporalSmooth maintains its own separate internal state
    And resetting one rule's state does not affect the other

  Scenario: State can be serialized and restored
    Given a stateful rule that has processed 50 frames
    When I serialize the rule's state to a file
    And I create a new instance of the same rule
    And I restore the state from the file
    Then the restored rule produces the same output for frame 51 as the original would have

  # --- Configuration ---

  Scenario: Load rule chain from YAML configuration
    Given a YAML configuration file defining a rule chain:
      """
      rules:
        - name: EdgeDetect
          params:
            algorithm: hed
            detail_level: 5
        - name: HalfToneRandomization
          params:
            dot_size: 4
            temporal_smoothing: 0.8
      """
    When I load and execute the rule chain
    Then the chain contains EdgeDetect followed by HalfToneRandomization
    And both rules use the specified parameters

  Scenario: Load rule chain from JSON configuration
    Given a JSON configuration file defining a rule chain
    When I load and execute the rule chain
    Then the chain is constructed and executed correctly
    And the result matches the equivalent YAML configuration

  Scenario: Override individual rule parameters via CLI
    Given a YAML configuration file with EdgeDetect at detail_level 5
    When I run the chain with a CLI override setting detail_level to 8
    Then EdgeDetect uses detail_level 8 instead of 5
    And all other parameters remain as specified in the configuration file

  # --- Rule Discovery and Registration ---

  Scenario: Pipeline discovers available rules at runtime
    Given the EffectRule plugin directory contains EdgeDetect, DepthMap, and HalfToneRandomization
    When I list available rules
    Then the list includes EdgeDetect, DepthMap, and HalfToneRandomization
    And each rule shows its name, description, and parameter schema

  Scenario: Custom rule is discovered after being added to plugin directory
    Given a custom EffectRule "MyCustomEffect" is placed in the plugin directory
    And the rule implements the EffectRule interface
    When I list available rules
    Then MyCustomEffect appears in the list
    And it can be used in a rule chain configuration

  # --- Layer System ---

  Scenario: Two rules read from the same named layer independently
    Given an input image "portrait.jpg"
    And a rule chain configured as:
      | name                    | input_layer | output_layer |
      | EdgeDetect              |             | edges        |
      | HalfToneRandomization   | edges       | halftone     |
      | AudioReactiveEdgeRule   | edges       | audio_lines  |
    When I run the rule chain
    Then HalfToneRandomization reads from the "edges" layer
    And AudioReactiveEdgeRule reads from the same "edges" layer
    And the "halftone" output is not affected by AudioReactiveEdgeRule processing
    And the "audio_lines" output is not affected by HalfToneRandomization processing

  Scenario: CompositeRule merges two layers with screen blend mode
    Given layers "halftone" and "audio_lines" have been produced by prior rules
    And a CompositeRule configured with:
      | layer       | blend_mode | opacity |
      | halftone    | screen     | 0.8     |
      | audio_lines | screen     | 1.0     |
    When I run the CompositeRule
    Then the output combines both layers using screen blending
    And the halftone layer is applied at 80% opacity
    And the audio_lines layer is applied at 100% opacity
    And both layers contribute to the final image without overwriting each other

  Scenario: Rule writes to a custom named layer
    Given an input image "portrait.jpg"
    And an EdgeDetect rule configured with output_layer "my_custom_edges"
    When I run the rule
    Then the edge detection output is stored in the "my_custom_edges" layer
    And the "main" layer is unchanged
    And subsequent rules can reference "my_custom_edges" as their input_layer

  Scenario: Rules without explicit layers default to main layer
    Given an input image "portrait.jpg"
    And a rule chain configured as [EdgeDetect, ThresholdBinarize] with no layer configuration
    When I run the rule chain
    Then EdgeDetect reads from "main" and writes to "main"
    And ThresholdBinarize reads from "main" and writes to "main"
    And the behavior is identical to standard linear chain processing

  Scenario: CompositeRule supports multiple blend modes
    Given layers "edges", "depth", and "halftone" have been produced by prior rules
    And a CompositeRule configured with:
      | layer    | blend_mode | opacity |
      | depth    | multiply   | 0.6     |
      | edges    | over       | 1.0     |
      | halftone | add        | 0.5     |
    When I run the CompositeRule
    Then each layer is composited using its specified blend mode
    And the compositing order follows the configured layer order (depth first, then edges, then halftone)

  Scenario: CompositeRule with background layer
    Given layers "halftone" and "audio_lines" have been produced by prior rules
    And a "depth" layer exists
    And a CompositeRule configured with background "depth" and foreground layers:
      | layer       | blend_mode | opacity |
      | halftone    | screen     | 0.8     |
      | audio_lines | add        | 1.0     |
    When I run the CompositeRule
    Then the depth layer is used as the base background
    And halftone and audio_lines are composited on top of it

  Scenario: Referencing a nonexistent layer produces clear error
    Given a rule chain where AudioReactiveEdgeRule has input_layer "nonexistent"
    When I attempt to load the rule chain
    Then the pipeline returns an error indicating layer "nonexistent" does not exist
    And the error lists available layer names at that point in the chain

  Scenario: Layer-based config via YAML
    Given a YAML configuration containing:
      """
      rules:
        - name: EdgeDetect
          output_layer: "edges"
          params:
            algorithm: hed
            detail_level: 5
        - name: HalfToneRandomization
          input_layer: "edges"
          output_layer: "halftone"
          params:
            dot_size: 4
        - name: AudioReactiveEdgeRule
          input_layer: "edges"
          output_layer: "audio_lines"
          params:
            sensitivity: 0.7
        - name: CompositeRule
          params:
            layers:
              - layer: "halftone"
                blend_mode: "screen"
                opacity: 0.8
              - layer: "audio_lines"
                blend_mode: "add"
                opacity: 1.0
            background: "depth"
      """
    When I load and execute the rule chain
    Then EdgeDetect writes to "edges"
    And both HalfToneRandomization and AudioReactiveEdgeRule read from "edges" independently
    And CompositeRule merges "halftone" and "audio_lines" over the "depth" background

  # --- Validation and Error Handling ---

  Scenario: Invalid rule name in configuration produces clear error
    Given a YAML configuration referencing a rule named "NonExistentRule"
    When I attempt to load the rule chain
    Then the pipeline returns an error indicating "NonExistentRule" is not a registered rule
    And the error lists available rule names

  Scenario: Invalid parameter for a rule produces clear error
    Given a YAML configuration with EdgeDetect and an unknown parameter "fake_param"
    When I attempt to load the rule chain
    Then the pipeline returns an error indicating "fake_param" is not a valid parameter for EdgeDetect
    And the error lists valid parameters for EdgeDetect

  Scenario: Rule chain validates before processing begins
    Given a YAML configuration with a rule chain
    When I load the rule chain
    Then the pipeline validates all rule names and parameters before processing any frames
    And if validation fails, no frames are processed

  Scenario: Rule processing error on one frame does not crash the pipeline
    Given a video with 100 frames
    And a rule chain that encounters a processing error on frame 50
    When I run the video pipeline
    Then the pipeline logs the error for frame 50
    And processing continues for frames 51-100
    And the output indicates which frames had errors

  Scenario: Empty rule chain produces clear error
    Given a YAML configuration with an empty rules list
    When I attempt to load the rule chain
    Then the pipeline returns an error indicating at least one rule is required
