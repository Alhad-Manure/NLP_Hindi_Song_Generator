import time
import os

try:
    from playsound import playsound
except ImportError:
    playsound = None

try:
    import winsound  # requires windows; only used if playsound/colab unavailable
except ImportError:
    winsound = None

try:
    from google.colab import output
except ImportError:
    output = None


def get_time_string():
    """
    Returns the current time as string in YYmmdd-HHMMSS format
    """
    return time.strftime("%Y%m%d-%H%M%S")


class AlertSounds:
    """
    Audio files available in the ptmlib/media directory
    """
    BEE5: str = 'media/bee5.mp3'
    DORE: str = 'media/dore.mp3'


class Stopwatch:

    """
    The Stopwatch class lets you measure the amount of time it takes to complete a long-running task.
    This can be useful for evaluating different machine learning models.
    """

    default_colab_sound_url: str = 'https://upload.wikimedia.org/wikipedia/commons/3/3b/Bee5th.ogg'

    def __init__(self):
        self._start_time = None

    def start(self) -> None:
        """
        Start the timer;
        Prints actual start time which is useful for long-running tasks

        :return: None
        """

        print("Start Time:", time.ctime())

        self._start_time = time.perf_counter()

    def stop(self, silent: bool = False, sound_path: str = AlertSounds.BEE5,
             colab_sound_url: str = default_colab_sound_url) -> None:

        """
        Stop the timer and print elapsed time info;
        Plays an audio prompt to alert that task has completed

        :param silent: disable audio alert
        :param sound_path: OPTIONS: ptmlib.time.AlertSounds value, or absolute file path to audio file
        :param colab_sound_url: absolute URL to audio file to be played if in Google Colab environment
        :return: None
        """

        if self._start_time is None:
            raise ValueError('start time must be set by calling start() before stop()')

        # stop and show duration immediately
        end_time = time.perf_counter()
        print("End Time:  ", time.ctime())
        print(f'Elapsed seconds: {end_time - self._start_time:0.4f}'
              + f' ({(end_time - self._start_time) / 60:0.2f} minutes)')

        # must reset
        self._start_time = None

        if not silent:
            # DO NOT allow audio error to impact processing
            try:
                self._alert_finished(sound_path, colab_sound_url)
            except Exception as ex:
                print('STOPWATCH _alert_finished ERROR:', ex)

    @staticmethod
    def _alert_finished(sound_path: str, colab_sound_url: str) -> None:

        dirname: str = ''

        try:
            dirname = os.path.dirname(__file__)
        except NameError:
            pass  # leave dirname as empty string; __file__ is not defined in colab

        # determine audio file to use
        if os.path.exists(sound_path):
            pass  # no changes needed
        elif os.path.exists(os.path.join(dirname, sound_path)):
            # relative path
            sound_path = os.path.join(dirname, sound_path)
        elif os.path.exists(os.path.join(dirname, str(AlertSounds.BEE5))):
            # switch back to default
            sound_path = os.path.join(dirname, str(AlertSounds.BEE5))
        else:
            sound_path = None

        if playsound is not None and sound_path is not None:
            # PLAYSOUND ON LOCAL
            playsound(sound_path)
        elif output is not None:
            # RUNNING IN COLAB
            # noinspection PyUnresolvedReferences
            output.eval_js(f'new Audio(\"{colab_sound_url}\").play()')
        elif winsound is not None:
            winsound.Beep(400, 1000)
