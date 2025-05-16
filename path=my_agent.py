def build_state(self, state: dict) -> torch.Tensor:
    bird_center_y = state['bird_y'] + (state['bird_height'] / 2)
    bird_velocity = state['bird_velocity']
    
    next_pipe_x = float('inf')
    next_pipe_width = float('inf')
    pipe_top_y = float('inf')
    pipe_bottom_y = float('inf')
    
    for pipe in state['pipes']:
        if pipe['x'] > state['bird_x'] - state['bird_width']:
            if pipe['x'] < next_pipe_x:
                next_pipe_x = pipe['x']
                next_pipe_width = pipe['width']
                pipe_top_y = pipe['top']
                pipe_bottom_y = pipe['bottom']
    
    # ... existing code ... 